/**
 * The `claude-code` pi provider, shown as "Anthropic": Claude models served by Claude Code
 * through the Agent SDK, with the user's Claude subscription, a `claude setup-token` token, or
 * an Anthropic API key. pi and OpenGUI see an ordinary provider; `runtime.ts` does the work.
 */
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  createAssistantMessageEventStream,
  createProvider,
  getCurrentSystemPrompt,
  getCurrentTools,
  type Api,
  type AssistantMessage,
  type Model,
  type OAuthAuth,
  type OAuthCredential,
  type Provider,
  type ProviderAuthInteraction,
  type SimpleStreamOptions,
  type ThinkingLevel,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { builtinProviders } from "@earendil-works/pi-ai/providers/all";
import { digest, errorMessage, transcriptEntries } from "./protocol.ts";
import {
  ClaudeCodeRuntime,
  decodeClaudeCodeCredential,
  encodeClaudeCodeCredential,
  writeJob,
  type ClaudeCodeCredential,
  type ClaudeCodeRuntimeOptions,
} from "./runtime.ts";

export const CLAUDE_CODE_PROVIDER_ID = "claude-code";
export const CLAUDE_CODE_API = "claude-code";

/** `claude setup-token` tokens and the local login do not expire on OpenGUI's side. */
const NEVER = Date.UTC(9999, 0, 1);

/** Models Claude Code serves, in menu order; metadata comes from pi's Anthropic catalog. */
const MODEL_IDS = [
  "claude-opus-5-5",
  "claude-sonnet-5-5",
  "claude-haiku-5-5",
  "claude-fable-5-2",
  "claude-fable-5-1",
  "claude-opus-5",
] as const;

/** Releases newer than the installed pi catalog: metadata of the nearest model, corrected. */
const NEWER_MODELS: Record<string, { from: string; name: string; cost: Model<Api>["cost"] }> = {
  "claude-fable-5-2": {
    from: "claude-fable-5-1",
    name: "Claude Fable 5.2",
    cost: { input: 8, output: 40, cacheRead: 0.2, cacheWrite: 10 },
  },
};

let catalog: Model<Api>[] | undefined;

/** The models of the `claude-code` provider. */
export function claudeCodeModels(): Model<Api>[] {
  if (catalog) return catalog;
  const anthropic = builtinProviders().find((provider) => provider.id === "anthropic");
  const known = new Map((anthropic?.getModels() ?? []).map((model) => [model.id, model]));
  catalog = MODEL_IDS.flatMap((id) => {
    const newer = NEWER_MODELS[id];
    const source = known.get(id) ?? (newer ? known.get(newer.from) : undefined);
    if (!source) return [];
    const { compat: _compat, headers: _headers, ...metadata } = source;
    return [
      {
        ...structuredClone(metadata),
        ...(known.has(id) || !newer ? {} : { name: newer.name, cost: newer.cost }),
        id,
        api: CLAUDE_CODE_API,
        provider: CLAUDE_CODE_PROVIDER_ID,
        baseUrl: "claude-code://local",
      } as Model<Api>,
    ];
  });
  return catalog;
}

const EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;
type Effort = (typeof EFFORTS)[number];

/** pi's thinking level as Claude Code's effort; no level keeps Claude Code's default. */
export function claudeCodeEffort(model: Model<Api>, level: ThinkingLevel | undefined) {
  if (!level) return undefined;
  const mapped = model.thinkingLevelMap?.[level];
  if (mapped === null) return undefined;
  const effort = mapped ?? level;
  if ((EFFORTS as readonly string[]).includes(effort)) return effort as Effort;
  return level === "minimal" ? "low" : undefined;
}

/**
 * How a subscription credential signs in to Claude Code:
 * - `account`: a Claude account signed in through Anthropic's OAuth flow in the browser. OpenGUI
 *   keeps its access and refresh tokens and hands Claude Code the current access token;
 * - `local`: the account the `claude` CLI is signed in to on the Host machine;
 * - `oauth-token` (or no mode, from before modes existed): a pasted `claude setup-token` token.
 */
type LoginMode = "account" | "local" | "oauth-token";
type ClaudeCodeOAuthCredential = OAuthCredential & { mode?: LoginMode };

function oauthToCredential(credential: ClaudeCodeOAuthCredential): ClaudeCodeCredential {
  return credential.mode === "local"
    ? { mode: "local" }
    : { mode: "oauth-token", token: credential.access };
}

/** pi's Anthropic OAuth flow; signs a Claude account in and refreshes its tokens. */
export type ClaudeAccountOAuth = Pick<OAuthAuth, "login" | "refresh">;

function piAnthropicOAuth(): ClaudeAccountOAuth {
  const oauth = builtinProviders().find((provider) => provider.id === "anthropic")?.auth.oauth;
  if (!oauth) throw new Error("This OpenGUI build has no Claude account sign-in");
  return oauth;
}

/**
 * pi's Anthropic flow either catches the browser's redirect on this machine or has the user paste
 * the code Anthropic shows. Only the desktop app's Host shares a machine with the browser.
 */
function accountLoginFlow(env: NodeJS.ProcessEnv = process.env) {
  return env.OPENGUI_SERVER_MODE === "desktop-sidecar" ? "browser" : "copy_code";
}

/** Whether `claude` is signed in on the Host machine; only then can its login be offered. */
export function claudeCodeLocalLoginAvailable(env: NodeJS.ProcessEnv = process.env) {
  // The desktop app shares the user's machine; macOS keeps the login in the Keychain.
  if (env.OPENGUI_SERVER_MODE === "desktop-sidecar") return true;
  const directory = env.CLAUDE_CONFIG_DIR?.trim() || join(homedir(), ".claude");
  return existsSync(join(directory, ".credentials.json"));
}

/** A request that started just before the token expired may run for a while: renew early. */
const ACCOUNT_RENEW_EARLY_MS = 30 * 60 * 1000;

/** The stored account credential, renewed well before its access token stops working. */
function accountCredential(
  next: OAuthCredential,
  previous?: OAuthCredential,
): ClaudeCodeOAuthCredential {
  const now = Date.now();
  const lifetime = Math.max(0, next.expires - now);
  return {
    ...next,
    type: "oauth",
    // Keep the refresh token when Anthropic does not rotate it.
    refresh: next.refresh || previous?.refresh || "",
    expires: now + Math.max(lifetime / 2, lifetime - ACCOUNT_RENEW_EARLY_MS),
    mode: "account",
  };
}

async function loginAccount(
  interaction: ProviderAuthInteraction,
  account: ClaudeAccountOAuth,
  flow: "browser" | "copy_code",
) {
  const credential = await account.login({
    ...interaction,
    // pi asks how to sign in; OpenGUI knows whether the browser shares the Host's machine.
    prompt: async (prompt) =>
      prompt.type === "select" && prompt.options.some((option) => option.id === flow)
        ? flow
        : interaction.prompt(prompt),
  });
  if (!credential.refresh) throw new Error("Anthropic returned no refresh token");
  return accountCredential(credential);
}

async function promptToken(interaction: ProviderAuthInteraction) {
  const token = (
    await interaction.prompt({
      type: "secret",
      message:
        "Run `claude setup-token` in a terminal, sign in with your Claude account, and paste the token it prints.",
      placeholder: "sk-ant-oat01-…",
      signal: interaction.signal,
    })
  ).trim();
  if (!token) throw new Error("No token was entered");
  return token;
}

export interface ClaudeCodeProviderOptions {
  /**
   * Whether this provider may use the account Claude Code is signed in to on this computer.
   * Only the Host's own scope may: a personal connection must bring its own credential.
   */
  allowLocalLogin?: boolean;
  /** Whether `claude` is signed in on this machine; checked when a sign-in starts. */
  localLoginAvailable?: () => boolean;
  /** The Claude account sign-in; pi's Anthropic OAuth flow by default. */
  account?: ClaudeAccountOAuth;
  /** How the account sign-in returns to OpenGUI; decided by where the Host runs by default. */
  accountLoginFlow?: () => "browser" | "copy_code";
  runtime?: ClaudeCodeRuntime;
}

let sharedRuntime: ClaudeCodeRuntime | undefined;

/** The process-wide runtime, shared by every scope's provider. Credentials never share a process. */
export function claudeCodeRuntime(options?: ClaudeCodeRuntimeOptions) {
  sharedRuntime ??= new ClaudeCodeRuntime({
    log:
      process.env.OPENGUI_CLAUDE_CODE_DEBUG === "1"
        ? (message, fields) =>
            process.stderr.write(
              `[OpenGUI Claude Code] ${message} ${JSON.stringify(fields ?? {})}\n`,
            )
        : undefined,
    ...options,
  });
  return sharedRuntime;
}

/** Replaces the process-wide runtime, e.g. with a fake Claude Code in tests. */
export async function setClaudeCodeRuntime(runtime: ClaudeCodeRuntime | undefined) {
  if (sharedRuntime && sharedRuntime !== runtime) await sharedRuntime.close();
  sharedRuntime = runtime;
}

/** Stops every Claude Code process this OpenGUI process started. */
export async function closeClaudeCodeSessions() {
  await sharedRuntime?.close();
}

function errorResponse(model: Model<Api>, message: string): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "error",
    errorMessage: message,
    timestamp: Date.now(),
  };
}

function errorStream(model: Model<Api>, message: string) {
  const stream = createAssistantMessageEventStream();
  const error = errorResponse(model, message);
  stream.push({ type: "error", reason: "error", error });
  stream.end(error);
  return stream;
}

/** Runtime failures are already worded for OpenGUI's error classification. */
function classified(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return /^Claude\b/u.test(message) ? message : errorMessage(message);
}

export function createClaudeCodeProvider(options: ClaudeCodeProviderOptions = {}): Provider {
  const allowLocalLogin = options.allowLocalLogin ?? true;
  const localLoginAvailable =
    options.localLoginAvailable ?? (() => claudeCodeLocalLoginAvailable());
  const account = () => options.account ?? piAnthropicOAuth();
  const loginFlow = options.accountLoginFlow ?? (() => accountLoginFlow());
  const runtime = () => options.runtime ?? claudeCodeRuntime();

  const streamSimple = (
    model: Model<Api>,
    context: TranscriptContext,
    streamOptions?: SimpleStreamOptions,
  ) => {
    let credential: ClaudeCodeCredential;
    try {
      credential = decodeClaudeCodeCredential(streamOptions?.apiKey);
    } catch (error) {
      return errorStream(model, classified(error));
    }
    if (credential.mode === "local" && !allowLocalLogin) {
      return errorStream(
        model,
        errorMessage("not logged in: this computer's Claude Code login is only for the Host"),
      );
    }
    const stream = createAssistantMessageEventStream();
    const started = Date.now();
    void (async () => {
      try {
        const systemPrompt = getCurrentSystemPrompt(context.messages);
        const tools = getCurrentTools(context.messages);
        const messages = transcriptEntries(context.messages);
        // OpenGUI passes a stable per-Session cache key; without one, the first turn identifies it.
        const key =
          streamOptions?.sessionId ??
          `anonymous:${digest({ systemPrompt, first: messages.slice(0, 1) })}`;
        const instance = runtime();
        const { job, session } = await instance.step({
          key,
          model,
          systemPrompt,
          tools,
          messages,
          effort: claudeCodeEffort(model, streamOptions?.reasoning),
          credential,
        });
        await writeJob({
          stream,
          job,
          model,
          signal: streamOptions?.signal,
          started,
          onAbort: () => instance.abort(session, job),
        });
      } catch (error) {
        const failed = errorResponse(model, classified(error));
        stream.push({ type: "error", reason: "error", error: failed });
        stream.end(failed);
      }
    })();
    return stream;
  };

  return createProvider({
    id: CLAUDE_CODE_PROVIDER_ID,
    name: "Anthropic",
    baseUrl: "claude-code://local",
    models: claudeCodeModels(),
    auth: {
      oauth: {
        name: "Claude Pro or Max plan (runs Claude Code)",
        loginLabel: "Sign in with Claude",
        isSubscription: true,
        async login(interaction) {
          const local = allowLocalLogin && localLoginAvailable();
          const mode = (await interaction.prompt({
            type: "select",
            message: "How should OpenGUI sign in to Claude?",
            options: [
              {
                id: "account",
                label: "Sign in with your Claude account",
                description:
                  "Opens claude.ai in your browser. Uses your Claude Pro or Max plan through Claude Code.",
              },
              ...(local
                ? [
                    {
                      id: "local",
                      label: "Use Claude Code's login on this computer",
                      description: "Uses the account the `claude` command is signed in to.",
                    },
                  ]
                : []),
              {
                id: "oauth-token",
                label: "Paste a token from `claude setup-token`",
                description: "A long-lived token for one Claude account.",
              },
            ],
            signal: interaction.signal,
          })) as LoginMode;
          if (mode === "account") return loginAccount(interaction, account(), loginFlow());
          if (mode === "local") {
            if (!local) throw new Error("Claude Code is not signed in on this computer");
            interaction.notify({
              type: "info",
              message:
                "OpenGUI uses the Claude account Claude Code is signed in to on this computer.",
            });
            return { type: "oauth", access: "local", refresh: "claude-code", expires: NEVER, mode };
          }
          if (mode !== "oauth-token") throw new Error(`Unknown sign-in method: ${String(mode)}`);
          const token = await promptToken(interaction);
          return { type: "oauth", access: token, refresh: "claude-code", expires: NEVER, mode };
        },
        async refresh(credential, signal) {
          if ((credential as ClaudeCodeOAuthCredential).mode !== "account") {
            return { ...credential, expires: NEVER };
          }
          return accountCredential(await account().refresh(credential, signal), credential);
        },
        async toAuth(credential) {
          const decoded = oauthToCredential(credential as ClaudeCodeOAuthCredential);
          return { apiKey: encodeClaudeCodeCredential(decoded) };
        },
      },
      apiKey: {
        name: "Anthropic API key (through Claude Code)",
        async login(interaction) {
          const key = (
            await interaction.prompt({
              type: "secret",
              message: "Anthropic API key. Requests are billed to its Console organization.",
              placeholder: "sk-ant-api03-…",
              signal: interaction.signal,
            })
          ).trim();
          if (!key) throw new Error("No API key was entered");
          return { type: "api_key", key };
        },
        async resolve({ credential }) {
          // Never ambient: the provider is configured only by a stored key.
          return credential?.key
            ? { auth: { apiKey: credential.key }, source: "Anthropic API key" }
            : undefined;
        },
      },
    },
    api: {
      stream: (model, context, streamOptions) => streamSimple(model, context, streamOptions),
      streamSimple,
    },
  });
}
