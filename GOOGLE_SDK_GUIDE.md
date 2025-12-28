# Google GenAI SDK 自定义参数使用指南

本文档基于 PixelLite 项目代码，说明如何在使用 Google GenAI SDK (`@google/genai`) 时自定义 `baseUrl` 等参数。

## 核心机制

后端 API (`api/analyze.ts` 和 `api/enhance.ts`) 采用了一套优先级逻辑来决定使用哪组配置：

1.  **用户自定义模式（高优先级）**：
    -   如果请求 Body 中包含 `apiKey`，系统将**强制**使用该 Key。
    -   在此模式下，系统会使用请求中传入的 `baseUrl`。如果未传入 `baseUrl`，则默认为 `undefined`（即使用 Google 官方默认地址）。
    -   **注意**：此模式下会完全忽略服务端的环境变量。

2.  **系统默认模式（低优先级）**：
    -   如果请求 Body 中**不包含** `apiKey`。
    -   系统将自动加载环境变量 `GEMINI_API_KEY` 和 `GEMINI_BASE_URL`。

## 后端实现细节

在后端 API 中，SDK 的初始化通过 `options` 对象进行配置。特别是 `baseUrl` 是通过 `httpOptions` 属性传入的。

### 代码参考 (`api/analyze.ts` / `api/enhance.ts`)

```typescript
// 1. 从请求体中解构参数
const { image, apiKey, model, baseUrl } = req.body;

let key: string | undefined;
let apiBaseUrl: string | undefined;

// 2. 确定配置来源
if (apiKey) {
    // 用户提供了 Key，使用用户提供的配置
    key = apiKey;
    apiBaseUrl = baseUrl; // 使用传入的 baseUrl (可能为 undefined)
} else {
    // 用户未提供 Key，使用环境变量
    key = process.env.GEMINI_API_KEY;
    apiBaseUrl = process.env.GEMINI_BASE_URL;
}

// 3. 构建 SDK 配置对象
const options: any = { apiKey: key };

// 如果存在 apiBaseUrl，则配置 httpOptions
if (apiBaseUrl) {
    // 关键点：通过 httpOptions.baseUrl 设置自定义请求地址
    // 这允许 SDK 指向自定义网关或代理地址
    options.httpOptions = { baseUrl: apiBaseUrl };
}

// 4. 实例化 SDK
const ai = new GoogleGenAI(options);
```

## 前端调用方式

前端服务 (`services/geminiService.ts`) 封装了 API 调用，允许将 `baseUrl` 透传给后端。

### 接口定义

`analyzeImage` 和 `generateEnhancedImage` 函数都接受 `baseUrl` 作为可选参数。

```typescript
// services/geminiService.ts

export const analyzeImage = async (
  base64Image: string,
  apiKey: string,
  baseUrl?: string, // <--- 可选的自定义 Base URL
  model: string = 'gemini-2.5-flash'
): Promise<AIData | null> => {
  // ... 发送 POST 请求到 /api/analyze
  // body: { image, apiKey, baseUrl, model }
};
```

### 调用示例

如果您需要使用自定义的 API 网关（例如 `https://my-custom-gateway.com`），可以按如下方式调用：

```typescript
import { analyzeImage, generateEnhancedImage } from './services/geminiService';

const myApiKey = "YOUR_CUSTOM_API_KEY";
const myBaseUrl = "https://my-custom-gateway.com"; // 自定义 Base URL

// 示例 1: 图片分析
const analysisResult = await analyzeImage(
    base64ImageString,
    myApiKey,
    myBaseUrl // 传入自定义 URL
);

// 示例 2: 图片增强
const enhancementResult = await generateEnhancedImage(
    base64ImageString,
    "Make it look like a painting",
    "gemini-2.5-flash-image",
    myApiKey,
    myBaseUrl // 传入自定义 URL
);
```

## 总结

要自定义 Google SDK 的行为（如修改请求地址），关键在于：

1.  **前端**：在调用 service 方法时显式传入 `baseUrl` 参数。
2.  **后端**：接收该参数，并在实例化 `GoogleGenAI` 时将其配置到 `httpOptions.baseUrl` 中。

这种设计允许应用灵活地在 "使用系统默认配置" 和 "用户自带 Key/Proxy" 之间切换，而无需修改后端代码。
