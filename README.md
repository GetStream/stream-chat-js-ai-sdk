# Stream Chat AI SDK

Shared utilities for building AI copilots that live inside Stream Chat channels. This package wires Stream’s real-time messaging primitives with Vercel’s AI SDK, tool calling, and optional Mem0 long-term memory so you can stand up production agents with minimal glue code.

## Features
- Connect Stream Chat channels to OpenAI, Anthropic, Gemini, or xAI via the Vercel AI SDK.
- Stream partial responses back to the channel with typing indicators and cancellation support.
- Register server-hosted tools (Node/TS functions) and propagate client tool definitions that get dispatched to front-end apps.
- Optional Mem0 memory layer (per user/channel) with zero extra code once environment variables are configured.
- Built-in helper to generate default tools (e.g., `getCurrentTemperature`) and to summarize conversations for channel lists.

## Installation

```bash
npm install @stream-io/chat-ai-sdk
# or
yarn add @stream-io/chat-ai-sdk
```

This library ships TypeScript types and transpiled JavaScript under `dist/`.

## Environment Variables

Set the following before instantiating an agent:

| Variable | Required | Description |
| --- | --- | --- |
| `STREAM_API_KEY` / `STREAM_API_SECRET` | ✅ | Stream server client used to upsert/connect the agent user. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`/`GEMINI_API_KEY`, `XAI_API_KEY` | ✅ (per provider) | API key for the provider selected via `AgentPlatform`. |
| `MEM0_API_KEY`, `MEM0_CONFIG_JSON`, `MEM0_DEFAULT_*` | Optional | Enables Mem0 memory when present (see `VercelAIAgent`). |
| `OPENWEATHER_API_KEY` | Optional | Needed only when you use the weather tool returned by `createDefaultTools`. |

## Quick Start

```ts
import {
  Agent,
  AgentPlatform,
  createDefaultTools,
  type ClientToolDefinition,
} from '@stream-io/chat-ai-sdk';

const agent = new Agent({
  userId: 'ai-bot-weather',
  channelId: 'support-room',
  platform: AgentPlatform.OPENAI,
  model: 'gpt-4o-mini',
  instructions: [
    'Answer in a friendly, concise tone.',
    'Prefer Celsius unless the user specifies otherwise.',
  ],
  serverTools: createDefaultTools(), // any AgentTool[]
  clientTools: [
    {
      name: 'openHelpCenter',
      description: 'Open the help center in the web app',
      parameters: {
        type: 'object',
        properties: { articleSlug: { type: 'string' } },
      },
    } satisfies ClientToolDefinition,
  ],
  mem0Context: {
    channelId: 'support-room',
    appId: 'stream-chat-support',
  },
});

await agent.start();

// Later:
await agent.stop();
```

### Tooling
- **Server tools** run in Node and receive the channel/message context plus typed arguments (validated by Zod). Register or replace them with `agent.registerServerTools`.
- **Client tools** are JSON-schema definitions that the SDK relays back to clients via Stream events so front-ends can execute privileged actions while maintaining the tool-call UX.

### Summaries

Need unread badges or channel list previews? Call `await agent.summarize(text)` or `Agent.generateSummary(text, platform)` to produce a short, LLM-generated headline.

## Build & Publish

```bash
npm run clean
npm run build
```

Publishing to npm is as simple as bumping the version and running:

```bash
npm publish --access public
```

## Project Structure

```
src/
├─ Agent.ts          // Manages lifecycle of the AI user inside Stream Chat
├─ VercelAIAgent.ts  // Handles streaming, tool invocation, and Mem0 integration
├─ defaultTools.ts   // Example AgentTool implementations
├─ serverClient.ts   // Stream server SDK bootstrap
└─ types.ts          // Shared enums/interfaces
```

Use `dist/` for the compiled output referenced by `package.json` (`main`/`types`).
