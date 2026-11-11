import type { AuthPrompt, OAuthCredential, ProviderAuthInteraction } from "@earendil-works/pi-ai";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test, vi } from "vite-plus/test";
import {
  claudeCodeLocalLoginAvailable,
  createClaudeCodeProvider,
  type ClaudeAccountOAuth,
} from "./provider.ts";
import { decodeClaudeCodeCredential } from "./runtime.ts";

const HOUR = 60 * 60 * 1000;
const directories: string[] = [];

afterEach(async () => {
  vi.useRealTimers();
  await Promise.all(
    directories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

/** Answers prompts in order and records what was asked. */
function interaction(answers: string[]) {
  const prompts: AuthPrompt[] = [];
  const events: unknown[] = [];
  const value: ProviderAuthInteraction = {
    signal: new AbortController().signal,
    prompt: async (prompt) => {
      prompts.push(prompt);
      const answer = answers.shift();
      if (answer === undefined) throw new Error(`Unexpected prompt: ${prompt.message}`);
      return answer;
    },
    notify: (event) => events.push(event),
  };
  return { value, prompts, events };
}

/** pi's Anthropic flow, without the network: asks its own select, then hands out tokens. */
function fakeAccount() {
  const selected: string[] = [];
  let refreshes = 0;
  const account: ClaudeAccountOAuth = {
    async login(flow) {
      selected.push(
        await flow.prompt({
          type: "select",
          message: "Select Anthropic login method:",
          options: [
            { id: "browser", label: "Browser login (default)" },
            { id: "copy_code", label: "Copy code login (headless)" },
          ],
        }),
      );
      flow.notify({ type: "auth_url", url: "https://claude.ai/oauth/authorize?code=true" });
      const code = await flow.prompt({ type: "manual_code", message: "Paste the code" });
      if (code !== "code#state") throw new Error("OAuth state mismatch");
      return {
        type: "oauth",
        access: "sk-ant-oat01-access",
        refresh: "refresh-1",
        expires: Date.now() + 8 * HOUR,
      };
    },
    async refresh(credential) {
      refreshes += 1;
      expect(credential.refresh).toBe("refresh-1");
      // Anthropic may answer without rotating the refresh token.
      return {
        type: "oauth",
        access: `sk-ant-oat01-access-${refreshes}`,
        refresh: "",
        expires: Date.now() + 8 * HOUR,
      };
    },
  };
  return { account, selected };
}

async function apiKeyOf(
  provider: ReturnType<typeof createClaudeCodeProvider>,
  credential: OAuthCredential,
) {
  const auth = await provider.auth.oauth!.toAuth(credential);
  return decodeClaudeCodeCredential(auth.apiKey);
}

describe("Claude Code provider sign-in", () => {
  test("is offered as Anthropic with a Claude plan sign-in", () => {
    const provider = createClaudeCodeProvider({ localLoginAvailable: () => false });
    expect(provider.name).toBe("Anthropic");
    expect(provider.auth.oauth).toMatchObject({
      isSubscription: true,
      loginLabel: "Sign in with Claude",
    });
  });

  test("signs a Claude account in through pi's flow and hands Claude Code its access token", async () => {
    vi.useFakeTimers({ now: Date.UTC(2026, 10, 11, 12) });
    const { account, selected } = fakeAccount();
    const provider = createClaudeCodeProvider({
      account,
      accountLoginFlow: () => "copy_code",
      localLoginAvailable: () => false,
    });
    const flow = interaction(["account", "code#state"]);
    const credential = await provider.auth.oauth!.login(flow.value);

    // OpenGUI answered pi's own select: the browser may be on another machine.
    expect(selected).toEqual(["copy_code"]);
    expect(flow.prompts.map((prompt) => prompt.type)).toEqual(["select", "manual_code"]);
    expect(flow.events).toContainEqual(expect.objectContaining({ type: "auth_url" }));
    expect(credential).toMatchObject({
      type: "oauth",
      access: "sk-ant-oat01-access",
      refresh: "refresh-1",
      mode: "account",
    });
    // Renewed half an hour before the token stops working.
    expect(credential.expires).toBe(Date.now() + 7.5 * HOUR);
    expect(await apiKeyOf(provider, credential)).toEqual({
      mode: "oauth-token",
      token: "sk-ant-oat01-access",
    });
  });

  test("the desktop app catches Anthropic's redirect on this machine", async () => {
    const { account, selected } = fakeAccount();
    const provider = createClaudeCodeProvider({
      account,
      accountLoginFlow: () => "browser",
      localLoginAvailable: () => false,
    });
    await provider.auth.oauth!.login(interaction(["account", "code#state"]).value);
    expect(selected).toEqual(["browser"]);
  });

  test("refreshes an account's tokens and keeps a refresh token Anthropic did not rotate", async () => {
    const { account } = fakeAccount();
    const provider = createClaudeCodeProvider({
      account,
      accountLoginFlow: () => "copy_code",
      localLoginAvailable: () => false,
    });
    const credential = await provider.auth.oauth!.login(
      interaction(["account", "code#state"]).value,
    );
    const refreshed = await provider.auth.oauth!.refresh!(credential, new AbortController().signal);
    expect(refreshed).toMatchObject({
      access: "sk-ant-oat01-access-1",
      refresh: "refresh-1",
      mode: "account",
    });
    expect(refreshed.expires).toBeGreaterThan(Date.now());
    expect(await apiKeyOf(provider, refreshed)).toEqual({
      mode: "oauth-token",
      token: "sk-ant-oat01-access-1",
    });
  });

  test("a pasted setup token never expires and is not refreshed with Anthropic", async () => {
    const refresh = vi.fn();
    const provider = createClaudeCodeProvider({
      account: { login: vi.fn(), refresh },
      localLoginAvailable: () => false,
    });
    const credential = await provider.auth.oauth!.login(
      interaction(["oauth-token", " sk-ant-oat01-setup "]).value,
    );
    expect(credential).toMatchObject({ access: "sk-ant-oat01-setup", mode: "oauth-token" });
    const refreshed = await provider.auth.oauth!.refresh!(credential, new AbortController().signal);
    expect(refreshed.expires).toBe(credential.expires);
    expect(refresh).not.toHaveBeenCalled();
  });

  test("offers the Host's Claude Code login only when `claude` is signed in there", async () => {
    const options = (flow: ReturnType<typeof interaction>) =>
      flow.prompts[0]?.type === "select" ? flow.prompts[0].options.map((option) => option.id) : [];

    const signedOut = interaction(["oauth-token", "sk-ant-oat01-setup"]);
    await createClaudeCodeProvider({ localLoginAvailable: () => false }).auth.oauth!.login(
      signedOut.value,
    );
    expect(options(signedOut)).toEqual(["account", "oauth-token"]);

    const signedIn = interaction(["local"]);
    const local = await createClaudeCodeProvider({
      localLoginAvailable: () => true,
    }).auth.oauth!.login(signedIn.value);
    expect(options(signedIn)).toEqual(["account", "local", "oauth-token"]);
    expect(local).toMatchObject({ mode: "local" });

    // A personal connection never gets the Host's login, signed in or not.
    const personal = interaction(["local"]);
    await expect(
      createClaudeCodeProvider({
        allowLocalLogin: false,
        localLoginAvailable: () => true,
      }).auth.oauth!.login(personal.value),
    ).rejects.toThrow("not signed in");
    expect(options(personal)).toEqual(["account", "oauth-token"]);
  });

  test("finds the `claude` login in its config directory", async () => {
    const config = await mkdtemp(join(tmpdir(), "opengui-claude-config-"));
    directories.push(config);
    expect(claudeCodeLocalLoginAvailable({ CLAUDE_CONFIG_DIR: config })).toBe(false);
    await writeFile(join(config, ".credentials.json"), "{}");
    expect(claudeCodeLocalLoginAvailable({ CLAUDE_CONFIG_DIR: config })).toBe(true);
    expect(claudeCodeLocalLoginAvailable({ OPENGUI_SERVER_MODE: "desktop-sidecar" })).toBe(true);
  });
});
