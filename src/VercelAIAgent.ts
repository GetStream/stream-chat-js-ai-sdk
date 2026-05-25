import type { ModelMessage, Tool } from 'ai';
import { streamText } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAI } from '@ai-sdk/openai';
import { createXai } from '@ai-sdk/xai';
import {
  createMem0,
  type Mem0ChatSettings,
  type Mem0ConfigSettings,
} from '@mem0/vercel-ai-provider';
import { z, type ZodType } from 'zod';
import type {
  Channel,
  Event,
  MessageResponse,
  StreamChat,
} from 'stream-chat';
import type { AIAgent } from './types';
import { AgentPlatform } from './types';

const BASE_SYSTEM_PROMPT =
  'You are an AI assistant. Help users with their questions. Evaluate each user turn independently: restate the user intent, decide whether it matches any available tool instructions, and only invoke the matching tool when the intent clearly applies. If no tool matches, answer normally.';
const CLIENT_TOOL_EVENT = 'custom_client_tool_invocation';

type IndicatorState =
  | 'AI_STATE_THINKING'
  | 'AI_STATE_GENERATING'
  | 'AI_STATE_EXTERNAL_SOURCES'
  | 'AI_STATE_ERROR';

type StreamTextOptions = Parameters<typeof streamText>[0];
type StreamLanguageModel = NonNullable<StreamTextOptions['model']>;

export interface Mem0ContextInput {
  userId?: string;
  channelId?: string;
  agentId?: string;
  appId?: string;
  configOverrides?: Partial<Mem0ConfigSettings>;
}

interface CreateModelOptions {
  mem0Context?: Mem0ContextInput;
  disableMem0?: boolean;
}

export interface ToolExecutionContext {
  channel: Channel;
  message: MessageResponse;
  sendEvent: (event: Record<string, unknown>) => Promise<void>;
}

export interface AgentTool {
  name: string;
  description: string;
  instructions?: string;
  parameters: ZodType;
  execute: (args: unknown, context: ToolExecutionContext) => Promise<string> | string;
  showExternalSourcesIndicator?: boolean;
}

export interface JsonSchemaDefinition {
  type?: string;
  description?: string;
  enum?: Array<string | number | boolean>;
  properties?: Record<string, JsonSchemaDefinition>;
  items?: JsonSchemaDefinition | JsonSchemaDefinition[];
  required?: string[];
  additionalProperties?: boolean | JsonSchemaDefinition;
  [key: string]: unknown;
}

export interface ClientToolDefinition {
  name: string;
  description: string;
  instructions?: string;
  parameters?: JsonSchemaDefinition;
  showExternalSourcesIndicator?: boolean;
}

export interface FinalMessageAugmentorContext {
  text: string;
  message: MessageResponse;
  channel: Channel;
  messages: ModelMessage[];
}

export type FinalMessageAugmentor = (
  context: FinalMessageAugmentorContext,
) => Promise<Record<string, unknown> | void> | Record<string, unknown> | void;

export interface UserMessageHandlerContext {
  message: MessageResponse;
  channel: Channel;
  chatClient: StreamChat;
}

export type UserMessageHandler = (
  context: UserMessageHandlerContext,
) => Promise<boolean> | boolean;

/**
 * A callback that converts an A2UI action into a natural-language description
 * that is appended to the user message sent to the LLM.
 *
 * Return a string to supply the description, or `undefined` to fall back to
 * the default behaviour (serialise the action name and raw context as JSON).
 */
export type A2uiActionProvider = (
  actionName: string,
  context: Record<string, unknown>,
) => string | undefined;

const cloneMem0ConfigOverrides = (
  config?: Partial<Mem0ConfigSettings>,
): Partial<Mem0ConfigSettings> | undefined => {
  if (!config) {
    return undefined;
  }
  const cloned: Partial<Mem0ConfigSettings> = { ...config };
  if (
    config.metadata &&
    typeof config.metadata === 'object' &&
    !Array.isArray(config.metadata)
  ) {
    cloned.metadata = { ...config.metadata };
  }
  return cloned;
};

const cloneMem0Context = (
  context?: Mem0ContextInput,
): Mem0ContextInput | undefined => {
  if (!context) {
    return undefined;
  }
  return {
    ...context,
    configOverrides: cloneMem0ConfigOverrides(context.configOverrides),
  };
};

const MEM0_CONFIG_FROM_ENV: Mem0ConfigSettings | undefined = (() => {
  const raw = process.env.MEM0_CONFIG_JSON;
  if (!raw) return undefined;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Mem0ConfigSettings;
    }
    console.warn('MEM0_CONFIG_JSON must be a JSON object. Ignoring value.');
  } catch (error) {
    console.warn('Failed to parse MEM0_CONFIG_JSON. Ignoring value.', error);
  }
  return undefined;
})();

const DEFAULT_MEM0_AGENT_ID =
  process.env.MEM0_DEFAULT_AGENT_ID?.trim() || 'stream-ai-agent';
const DEFAULT_MEM0_APP_ID =
  process.env.MEM0_DEFAULT_APP_ID?.trim() || 'stream-ai-app';

const buildMem0Settings = (context?: Mem0ContextInput): Mem0ChatSettings => {
  const envDefaults = MEM0_CONFIG_FROM_ENV
    ? { ...MEM0_CONFIG_FROM_ENV }
    : {};
  const overrides = context?.configOverrides
    ? { ...context.configOverrides }
    : {};
  const merged: Mem0ConfigSettings = {
    ...envDefaults,
    ...overrides,
  };

  const envUserId = process.env.MEM0_DEFAULT_USER_ID?.trim();
  const envAgentId = process.env.MEM0_DEFAULT_AGENT_ID?.trim();
  const envAppId = process.env.MEM0_DEFAULT_APP_ID?.trim();
  const fallbackChannel = context?.channelId;

  merged.user_id =
    context?.userId ??
    overrides.user_id ??
    merged.user_id ??
    envUserId ??
    fallbackChannel ??
    context?.agentId;

  merged.agent_id =
    context?.agentId ??
    overrides.agent_id ??
    merged.agent_id ??
    envAgentId ??
    context?.userId;

  merged.app_id =
    context?.appId ??
    overrides.app_id ??
    merged.app_id ??
    envAppId ??
    fallbackChannel;

  return merged;
};

const getMem0ProviderName = (platform: AgentPlatform): string | null => {
  switch (platform) {
    case AgentPlatform.OPENAI:
      return 'openai';
    case AgentPlatform.ANTHROPIC:
      return 'anthropic';
    case AgentPlatform.GEMINI:
      return 'google';
    default:
      return null;
  }
};

const tryCreateMem0Model = (
  platform: AgentPlatform,
  providerApiKey: string,
  modelId: string,
  context?: Mem0ContextInput,
): StreamLanguageModel | undefined => {
  const mem0ApiKey = process.env.MEM0_API_KEY?.trim();
  if (!mem0ApiKey) {
    return undefined;
  }
  const providerName = getMem0ProviderName(platform);
  if (!providerName) {
    return undefined;
  }

  try {
    const mem0Settings = buildMem0Settings(context);
    const mem0Provider = createMem0({
      provider: providerName,
      apiKey: providerApiKey,
      mem0ApiKey,
      mem0Config: mem0Settings,
    });

    return mem0Provider(modelId, mem0Settings) as unknown as StreamLanguageModel;
  } catch (error) {
    console.warn(
      `Mem0 integration failed for platform ${platform}. Falling back to base model.`,
      (error as Error).message,
    );
    return undefined;
  }
};

export const createModelForPlatform = (
  platform: AgentPlatform,
  modelOverride?: string,
  options?: CreateModelOptions,
): StreamLanguageModel => {
  const modelId = typeof modelOverride === 'string' && modelOverride.trim()
    ? modelOverride.trim()
    : undefined;
  switch (platform) {
    case AgentPlatform.OPENAI: {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error('OpenAI API key is required');
      }
      const resolvedModelId = modelId ?? 'gpt-4o-mini';
      const mem0Model =
        options?.disableMem0
          ? undefined
          : tryCreateMem0Model(
            platform,
            apiKey,
            resolvedModelId,
            options?.mem0Context,
          );
      if (mem0Model) {
        return mem0Model;
      }
      const openai = createOpenAI({ apiKey });
      return openai(resolvedModelId) as StreamLanguageModel;
    }
    case AgentPlatform.ANTHROPIC: {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        throw new Error('Anthropic API key is required');
      }
      const resolvedModelId = modelId ?? 'claude-haiku-4-5';
      const mem0Model =
        options?.disableMem0
          ? undefined
          : tryCreateMem0Model(
            platform,
            apiKey,
            resolvedModelId,
            options?.mem0Context,
          );
      if (mem0Model) {
        return mem0Model;
      }
      const anthropic = createAnthropic({ apiKey });
      return anthropic(resolvedModelId) as StreamLanguageModel;
    }
    case AgentPlatform.GEMINI: {
      const apiKey =
        process.env.GOOGLE_GENERATIVE_AI_API_KEY ??
        process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error('Gemini API key is required');
      }
      const resolvedModelId = modelId ?? 'gemini-1.5-flash';
      const mem0Model =
        options?.disableMem0
          ? undefined
          : tryCreateMem0Model(
            platform,
            apiKey,
            resolvedModelId,
            options?.mem0Context,
          );
      if (mem0Model) {
        return mem0Model;
      }
      const google = createGoogleGenerativeAI({ apiKey });
      return google(resolvedModelId) as StreamLanguageModel;
    }
    case AgentPlatform.XAI: {
      const apiKey = process.env.XAI_API_KEY;
      if (!apiKey) {
        throw new Error('xAI API key is required');
      }
      const xai = createXai({ apiKey });
      return xai((modelId ?? 'grok-beta')) as StreamLanguageModel;
    }
    default:
      throw new Error(`Unsupported AI platform: ${platform}`);
  }
};

export class VercelAIAgent implements AIAgent {
  private lastInteractionTs = Date.now();
  private handlers = new Set<VercelResponseHandler>();
  private serverTools: AgentTool[];
  private clientTools: AgentTool[] = [];
  private readonly modelOverride?: string;
  private readonly additionalInstructions: string[];
  private readonly mem0Context?: Mem0ContextInput;
  private readonly finalMessageAugmentor?: FinalMessageAugmentor;
  private readonly userMessageHandler?: UserMessageHandler;
  private readonly a2uiActionProvider?: A2uiActionProvider;

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel,
    private readonly platform: AgentPlatform,
    tools: AgentTool[] = [],
    modelOverride: string | undefined,
    additionalInstructions: string[] | undefined,
    mem0Context: Mem0ContextInput | undefined,
    private serverClient: StreamChat,
    finalMessageAugmentor?: FinalMessageAugmentor,
    userMessageHandler?: UserMessageHandler,
    a2uiActionProvider?: A2uiActionProvider,
  ) {
    this.serverTools = tools ?? [];
    this.modelOverride = modelOverride;
    this.additionalInstructions = Array.isArray(additionalInstructions)
      ? additionalInstructions.filter((line) => line && line.trim().length)
      : additionalInstructions
        ? [additionalInstructions]
        : [];
    this.mem0Context = mem0Context;
    this.finalMessageAugmentor = finalMessageAugmentor;
    this.userMessageHandler = userMessageHandler;
    this.a2uiActionProvider = a2uiActionProvider;
  }

  init = async () => {
    // Ensure model configuration is valid before listening for messages.
    this.createLanguageModel();
    this.chatClient.on('message.new', this.handleMessage);
  };

  dispose = async () => {
    this.chatClient.off('message.new', this.handleMessage);
    await this.chatClient.disconnectUser();

    const handlers = Array.from(this.handlers);
    this.handlers.clear();
    await Promise.allSettled(handlers.map((handler) => handler.dispose()));
  };

  getLastInteraction = (): number => this.lastInteractionTs;

  setServerTools = (tools: AgentTool[]) => {
    this.serverTools = tools ?? [];
  };

  addServerTools = (tools: AgentTool[]) => {
    this.serverTools = [...this.serverTools, ...(tools ?? [])];
  };

  setClientTools = (tools: AgentTool[]) => {
    this.clientTools = tools ?? [];
  };

  setClientToolDefinitions = (definitions: ClientToolDefinition[]) => {
    const tools = (definitions ?? []).map((definition) =>
      this.createClientTool(definition),
    );
    this.setClientTools(tools);
  };

  private createLanguageModel(userId?: string): StreamLanguageModel {
    return createModelForPlatform(this.platform, this.modelOverride, {
      mem0Context: this.buildMem0ContextForUser(userId),
    });
  }

  private buildMem0ContextForUser(
    userId?: string,
  ): Mem0ContextInput | undefined {
    let base = cloneMem0Context(this.mem0Context);
    if (!base && this.channel?.id) {
      base = {
        channelId: this.channel.id,
      };
    }
    if (!base && !userId) {
      return undefined;
    }
    const context: Mem0ContextInput = base ?? {};
    if (!context.agentId) {
      context.agentId = DEFAULT_MEM0_AGENT_ID;
    }
    if (!context.appId) {
      context.appId = DEFAULT_MEM0_APP_ID;
    }
    if (!context.channelId && this.channel?.id) {
      context.channelId = this.channel.id;
    }
    if (userId) {
      context.userId = userId;
      const metadata: Record<string, any> = {
        ...(context.configOverrides?.metadata ?? {}),
        user_id: userId,
      };
      context.configOverrides = {
        ...(context.configOverrides ?? {}),
        metadata,
      };
    }
    return context;
  }

  private getActiveTools(): AgentTool[] {
    return [...this.serverTools, ...this.clientTools];
  }

  private createClientTool(definition: ClientToolDefinition): AgentTool {
    const parameters =
      jsonSchemaToZod(definition.parameters) ?? z.object({}).passthrough();
    return {
      name: definition.name,
      description: definition.description,
      instructions: definition.instructions,
      parameters,
      showExternalSourcesIndicator: definition.showExternalSourcesIndicator,
      execute: async (args, context) => {
        console.log(
          `[ClientTool] Dispatching ${definition.name} with args:`,
          args,
        );
        await context.sendEvent({
          type: CLIENT_TOOL_EVENT,
          cid: context.message.cid,
          message_id: context.message.id,
          channel_id: context.channel.id,
          channel_type: context.channel.type,
          tool: {
            name: definition.name,
            description: definition.description,
            instructions: definition.instructions,
            parameters: definition.parameters ?? null,
          },
          args: args ?? {},
        });
        return `Client tool "${definition.name}" invocation dispatched.`;
      },
    };
  }

  private getSystemPrompt(): string {
    const instructions = [
      ...this.additionalInstructions,
      ...this.getActiveTools()
        .map((tool) => tool.instructions ?? '')
        .filter((value) => typeof value === 'string' && value.trim().length > 0),
    ]
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (!instructions.length) {
      return BASE_SYSTEM_PROMPT;
    }

    const formatted = instructions.map((line) => `- ${line}`).join('\n');
    return `${BASE_SYSTEM_PROMPT}\n\nGuidelines:\n${formatted}`;
  }

  private handleMessage = async (event: Event) => {
    const incomingMessage = event.message as MessageResponse & {
      ai_generated?: boolean;
    };
    if (!incomingMessage || incomingMessage.ai_generated) {
      console.log(`[handleMessage] skipping — no message or ai_generated`);
      return;
    }

    if (this.userMessageHandler) {
      const handled = await this.userMessageHandler({
        message: incomingMessage,
        channel: this.channel,
        chatClient: this.chatClient,
      });
      if (handled) {
        return;
      }
    }

    let incomingText = incomingMessage.text?.trim();
    const a2uiQuery = this.extractA2uiActionQuery(incomingMessage);
    if (a2uiQuery) {
      incomingText = a2uiQuery;
    }
    if (!incomingText) {
      return;
    }

    const userId = incomingMessage.user?.id;
    if (!userId || userId.startsWith('ai-bot')) return;

    this.lastInteractionTs = Date.now();

    const history = this.channel.state.messages
      .slice(-10)
      .filter((msg) => msg.text && msg.text.trim() !== '')
      .map<ModelMessage>((msg) => ({
        role: msg.user?.id.startsWith('ai-bot') ? 'assistant' : 'user',
        content: msg.text ?? '',
      }));

    const lastHistoryEntry = history[history.length - 1];
    if (
      lastHistoryEntry?.content !== incomingText ||
      lastHistoryEntry.role !== 'user'
    ) {
      history.push({ role: 'user', content: incomingText });
    }

    const messages: ModelMessage[] = [
      { role: 'system', content: this.getSystemPrompt() },
      ...history,
    ];

    const { message: channelMessage } = await this.channel.sendMessage({
      text: '',
      ai_generated: true,
    } as any);

    await this.safeSendEvent({
      type: 'ai_indicator.update',
      ai_state: 'AI_STATE_THINKING',
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    let languageModel: StreamLanguageModel;
    try {
      languageModel = this.createLanguageModel(userId);
    } catch (error) {
      console.error('Failed to initialize language model', error);
      await this.handlePreflightError(channelMessage, error as Error);
      return;
    }

    const handler = new VercelResponseHandler(
      languageModel,
      this.chatClient,
      this.channel,
      channelMessage,
      messages,
      (event) => this.safeSendEvent(event),
      () => this.getActiveTools(),
      this.serverClient,
      this.finalMessageAugmentor,
    );

    this.handlers.add(handler);
    void handler
      .run()
      .catch((error) => {
        console.error('AI handler error', error);
      })
      .finally(() => {
        this.handlers.delete(handler);
      });
  };

  private extractA2uiActionQuery(
    message: MessageResponse & {
      extraData?: Record<string, unknown>;
    },
  ): string | undefined {
    const payload =
      this.extractStringValue(message.extraData?.['a2ui_interaction']) ??
      this.extractStringValue(
        (message as unknown as Record<string, unknown>)['a2ui_interaction'],
      );
    if (!payload) {
      return undefined;
    }

    try {
      const parsed = JSON.parse(payload);
      const userAction = parsed?.userAction;
      if (!userAction || typeof userAction !== 'object') {
        return undefined;
      }
      const actionName = this.extractStringValue(
        (userAction as Record<string, unknown>).name,
      );
      const context = this.extractObjectValue(
        (userAction as Record<string, unknown>).context,
      );
      return this.describeA2uiAction(actionName, context);
    } catch (error) {
      console.warn('Failed to parse a2ui_interaction payload', error);
      return undefined;
    }
  }

  private describeA2uiAction(
    actionName?: string,
    context?: Record<string, unknown>,
  ): string | undefined {
    if (!actionName) {
      return undefined;
    }
    const ctx = context ?? {};

    if (this.a2uiActionProvider) {
      const result = this.a2uiActionProvider(actionName, ctx);
      if (result !== undefined) {
        return result;
      }
    }

    return `User submitted an event: ${actionName} with data: ${JSON.stringify(
      ctx,
    )}`;
  }

  private extractContextValue(value: unknown): string | undefined {
    if (value == null) {
      return undefined;
    }
    if (typeof value === 'string') {
      return value;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
    if (Array.isArray(value)) {
      return value
        .map((item) => this.extractContextValue(item))
        .filter((item): item is string => Boolean(item))
        .join(', ');
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return undefined;
  }

  private extractStringValue(value: unknown): string | undefined {
    if (typeof value !== 'string') {
      return undefined;
    }
    const trimmed = value.trim();
    return trimmed.length ? trimmed : undefined;
  }

  private extractObjectValue(
    value: unknown,
  ): Record<string, unknown> | undefined {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
    return undefined;
  }

  private async handlePreflightError(
    message: MessageResponse,
    error: Error,
  ): Promise<void> {
    await this.safeSendEvent({
      type: 'ai_indicator.update',
      ai_state: 'AI_STATE_ERROR',
      cid: message.cid,
      message_id: message.id,
    });
    await this.chatClient.partialUpdateMessage(message.id, {
      set: {
        text: error.message ?? 'Error generating the message',
        generating: false,
      } as any,
    });
  }

  private async safeSendEvent(event: Record<string, unknown>) {
    const maxAttempts = 5;
    let delay = 100;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        await this.channel.sendEvent(event as any);
        return;
      } catch (err) {
        const status = (err as any)?.status || (err as any)?.response?.status;
        const retryable = status === 429 || (status >= 500 && status < 600);
        if (!retryable || attempt === maxAttempts) {
          const label = retryable
            ? 'Failed to send event after retries'
            : 'Failed to send event';
          console.error(label, err);
          return;
        }
        await new Promise((resolve) =>
          setTimeout(resolve, delay + Math.floor(Math.random() * 50)),
        );
        delay *= 2;
      }
    }
  }

}

class VercelResponseHandler {
  private controller: AbortController | null = null;
  private messageText = '';
  private finalized = false;
  private aborted = false;
  private currentIndicator?: IndicatorState;
  private indicatorCleared = false;
  private pendingUpdateTimer: ReturnType<typeof setTimeout> | null = null;
  private lastUpdatePromise: Promise<void> = Promise.resolve();
  private readonly updateIntervalMs = 200;
  private disposed = false;
  private readonly toolsResolver: () => AgentTool[];
  private readonly finalMessageAugmentor?: FinalMessageAugmentor;

  constructor(
    private readonly model: StreamLanguageModel,
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly messages: ModelMessage[],
    private readonly sendEvent: (event: Record<string, unknown>) => Promise<void>,
    toolsResolver: () => AgentTool[],
    private readonly serverClient: StreamChat,
    finalMessageAugmentor?: FinalMessageAugmentor,
  ) {
    this.chatClient.on('ai_indicator.stop', this.handleStopGenerating);
    this.toolsResolver = toolsResolver;
    this.finalMessageAugmentor = finalMessageAugmentor;
  }

  run = async () => {
    this.controller = new AbortController();

    try {
      const toolDefinitions = this.buildToolDefinitions();
      const streamOptions = {
        model: this.model,
        messages: this.messages,
        abortSignal: this.controller.signal,
      } as Parameters<typeof streamText>[0];
      if (toolDefinitions) {
        (streamOptions as any).toolChoice = 'auto';
        (streamOptions as any).tools = toolDefinitions;
      }

      const result = await streamText(streamOptions);

      await this.consumeStream(result);
      if (!this.aborted) {
        await this.finalizeMessage();
      }
    } catch (error) {
      if (this.aborted) {
        await this.finalizeMessage();
      } else {
        await this.handleError(error as Error);
      }
    } finally {
      await this.dispose();
    }
  };

  private buildToolDefinitions(): Record<string, Tool<any, any>> | null {
    const tools = this.toolsResolver();
    if (!tools.length) {
      return null;
    }
    return tools.reduce<Record<string, Tool<any, any>>>(
      (acc, tool) => {
        const toolDefinition: Tool<any, any> = {
          description: tool.description,
          inputSchema: tool.parameters as any,
          execute: async (args: unknown) => {
            if (tool.showExternalSourcesIndicator !== false) {
              await this.updateIndicator('AI_STATE_EXTERNAL_SOURCES');
            }
            const result = await tool.execute(args, {
              channel: this.channel,
              message: this.message,
              sendEvent: (event) => this.sendEvent(event),
            });
            if (typeof result === 'string') {
              return result;
            }
            return JSON.stringify(result);
          },
        };
        acc[tool.name] = toolDefinition;
        return acc;
      },
      {},
    );
  }

  dispose = async () => {
    if (this.disposed) return;
    this.disposed = true;
    this.chatClient.off('ai_indicator.stop', this.handleStopGenerating);
    if (this.pendingUpdateTimer) {
      clearTimeout(this.pendingUpdateTimer);
      this.pendingUpdateTimer = null;
    }
  };

  private async consumeStream(result: any) {
    const fullStream = result?.fullStream;
    if (fullStream) {
      for await (const payload of fullStream) {
        const part = (payload && typeof payload === 'object' && 'part' in payload
          ? (payload as { part?: { type?: string; [key: string]: unknown } }).part
          : payload) as { type?: string; text?: string; textDelta?: string; output?: string; error?: unknown } | null;
        if (!part || !part.type) {
          continue;
        }

        switch (part.type) {
          case 'text-delta': {
            const delta = part.text ?? (part as any).textDelta;
            if (!delta) break;
            await this.updateIndicator('AI_STATE_GENERATING');
            this.messageText += delta;
            this.schedulePartialUpdate();
            break;
          }
          case 'tool-result': {
            const toolOutput = part.output;
            if (!toolOutput) break;
            this.messageText += toolOutput;
            this.schedulePartialUpdate();
            break;
          }
          case 'error': {
            const error =
              part.error instanceof Error
                ? part.error
                : new Error(
                    typeof part.error === 'string'
                      ? part.error
                      : 'Error streaming response',
                  );
            throw error;
          }
          case 'finish':
          case 'abort':
            return;
          default:
            // ignore other chunk types (response metadata, tool events, etc.)
            break;
        }
      }
      return;
    }

    const textStream = result?.textStream;
    if (!textStream) return;
    for await (const delta of textStream) {
      if (!delta) continue;
      await this.updateIndicator('AI_STATE_GENERATING');
      this.messageText += delta;
      this.schedulePartialUpdate();
    }
  }

  private handleStopGenerating = async (event: Event) => {
    const messageId = (event as unknown as { message_id?: string })?.message_id;
    if (messageId && messageId !== this.message.id) {
      return;
    }

    this.aborted = true;
    try {
      this.controller?.abort();
    } catch (e) {
      // no-op
    }
  };

  private schedulePartialUpdate() {
    if (this.finalized) return;
    if (this.pendingUpdateTimer) return;
    this.pendingUpdateTimer = setTimeout(() => {
      this.pendingUpdateTimer = null;
      void this.flushPartialUpdate();
    }, this.updateIntervalMs);
  }

  private async flushPartialUpdate() {
    if (this.finalized) return;
    const text = this.messageText;
    const id = this.message.id;
    const userId = this.chatClient.user?.id;
    this.lastUpdatePromise = this.lastUpdatePromise.then(() => {
      return this.serverClient
        .ephemeralUpdateMessage(
          id,
          { set: { text, generating: true } as any },
          userId,
        )
        .then(
          () => console.log(`[ephemeralUpdateMessage] success messageId=${id}`),
          (err) => console.error("[ephemeralUpdateMessage] failed", err),
        );
    });
    await this.lastUpdatePromise;
  }

  private async updateIndicator(state: IndicatorState) {
    if (this.currentIndicator === state) return;
    this.currentIndicator = state;
    this.indicatorCleared = false;
    await this.sendEvent({
      type: 'ai_indicator.update',
      ai_state: state,
      cid: this.message.cid,
      message_id: this.message.id,
    });
  }

  private async clearIndicator() {
    if (this.indicatorCleared) return;
    this.currentIndicator = undefined;
    this.indicatorCleared = true;
    await this.sendEvent({
      type: 'ai_indicator.clear',
      cid: this.message.cid,
      message_id: this.message.id,
    });
  }

  private async buildFinalSetPayload(): Promise<Record<string, unknown>> {
    const basePayload: Record<string, unknown> = {
      text: this.messageText,
      generating: false,
    };
    if (!this.finalMessageAugmentor) {
      return basePayload;
    }
    try {
      const augmented = await this.finalMessageAugmentor({
        text: this.messageText,
        message: this.message,
        channel: this.channel,
        messages: this.messages,
      });
      if (augmented && typeof augmented === 'object') {
        return { ...basePayload, ...augmented };
      }
    } catch (error) {
      console.error('Final message augmentor failed', error);
    }
    return basePayload;
  }

  private async finalizeMessage() {
    if (this.finalized) return;
    this.finalized = true;
    if (this.pendingUpdateTimer) {
      clearTimeout(this.pendingUpdateTimer);
      this.pendingUpdateTimer = null;
    }
    await this.lastUpdatePromise.catch(() => Promise.resolve());
    const setPayload = await this.buildFinalSetPayload();
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: setPayload as any,
    });
    await this.clearIndicator();
  }

  private async handleError(error: Error) {
    this.finalized = true;
    await this.sendEvent({
      type: 'ai_indicator.update',
      ai_state: 'AI_STATE_ERROR',
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: error.message ?? 'Error generating the message',
        generating: false,
      } as any,
    });
  }
}

const jsonSchemaToZod = (schema?: JsonSchemaDefinition): ZodType => {
  if (!schema) {
    return z.looseObject({});
  }

  if (
    Array.isArray(schema.enum) &&
    schema.enum.length > 0 &&
    schema.enum.every((value) => typeof value === 'string')
  ) {
    const enumValues = schema.enum as string[];
    const enumSchema =
      enumValues.length === 1
        ? z.literal(enumValues[0])
        : z.enum(enumValues as [string, ...string[]]);
    return applyDescription(enumSchema, schema.description);
  }

  const inferredType = schema.type ?? (schema.properties ? 'object' : undefined);

  switch (inferredType) {
    case 'string':
      return applyDescription(z.string(), schema.description);
    case 'number':
      return applyDescription(z.number(), schema.description);
    case 'integer':
      return applyDescription(z.number().int(), schema.description);
    case 'boolean':
      return applyDescription(z.boolean(), schema.description);
    case 'array': {
      const itemSchemas = Array.isArray(schema.items)
        ? schema.items
        : schema.items
          ? [schema.items]
          : [];
      const firstItemSchema = itemSchemas[0] as JsonSchemaDefinition | undefined;
      const itemZod = firstItemSchema
        ? jsonSchemaToZod(firstItemSchema)
        : z.any();
      return applyDescription(z.array(itemZod), schema.description);
    }
    case 'object':
    default: {
      const properties = schema.properties ?? {};
      const required = new Set(schema.required ?? []);
      const shape: Record<string, ZodType> = {};
      for (const [key, propertySchema] of Object.entries(properties)) {
        let fieldSchema = jsonSchemaToZod(propertySchema);
        if (!required.has(key)) {
          fieldSchema = fieldSchema.optional();
        }
        shape[key] = fieldSchema;
      }

      let objectSchema: any = z.object(shape);
      const additional = schema.additionalProperties;
      if (typeof additional === 'object') {
        objectSchema = objectSchema.catchall(jsonSchemaToZod(additional));
      } else if (additional === false) {
        objectSchema = objectSchema.strict();
      } else {
        objectSchema = objectSchema.passthrough();
      }

      return applyDescription(objectSchema as ZodType, schema.description);
    }
  }
};

const applyDescription = <T extends ZodType>(
  schema: T,
  description?: string,
): T => {
  if (description) {
    return schema.describe(description) as T;
  }
  return schema;
};
