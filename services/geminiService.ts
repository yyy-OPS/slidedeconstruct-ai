import { GoogleGenAI, Type, GenerateContentResponse, HarmCategory, HarmBlockThreshold, Schema } from "@google/genai";
import { SlideAnalysisResult, ElementType, AISettings, DEFAULT_AI_SETTINGS, ProviderConfig, ReconstructedSlideResult, PPTShapeElement, BoundingBox, SlideVisualElement, SlideTextElement, PPTShapeType } from "../types";

// State to hold current settings
let currentSettings: AISettings = { ...DEFAULT_AI_SETTINGS };

// Allow initializing with env var if available and no user setting
// Note: In Vite, environment variables need to be prefixed with VITE_ to be accessible in browser


export const updateSettings = (settings: AISettings) => {
    currentSettings = settings;
};

export const getSettings = () => currentSettings;

// --- Helpers ---

const cleanJsonString = (str: string): string => {
    if (!str) return "{}";
    let cleaned = str.trim();

    // Remove markdown wrapping if present
    if (cleaned.startsWith('```json')) {
        cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (cleaned.startsWith('```')) {
        cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    // Attempt to extract JSON object if there's extra text
    const firstOpen = cleaned.indexOf('{');
    const lastClose = cleaned.lastIndexOf('}');
    if (firstOpen !== -1 && lastClose !== -1 && lastClose > firstOpen) {
        cleaned = cleaned.substring(firstOpen, lastClose + 1);
    }

    return cleaned;
};

/**
 * Robustly try to parse JSON, attempting to repair truncated strings if necessary.
 */
const tryParseJSON = (jsonStr: string): any => {
    try {
        return JSON.parse(jsonStr);
    } catch (e: any) {
        console.warn("JSON Parse failed, attempting repair. Raw:", jsonStr);
        let repaired = jsonStr.trim();

        if (repaired.lastIndexOf(']') === -1 || repaired.lastIndexOf(']') < repaired.lastIndexOf('[')) {
            if (repaired.endsWith(',')) {
                repaired = repaired.slice(0, -1);
            }
            if (repaired.endsWith('}')) {
                repaired += ']}';
            } else {
                repaired += ']}';
            }
        } else if (repaired.lastIndexOf('}') === -1) {
            repaired += '}';
        }

        try {
            return JSON.parse(repaired);
        } catch (e2) {
            console.error("JSON Repair failed:", e2);
            throw new Error(`JSON Parse Error: ${e.message}. \nRaw: ${jsonStr.substring(0, 100)}...`);
        }
    }
}

const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

const findElementsArray = (obj: any): any[] => {
    if (!obj) return [];
    if (Array.isArray(obj)) return obj;
    if (Array.isArray(obj.elements)) return obj.elements;
    if (Array.isArray(obj.items)) return obj.items;
    if (Array.isArray(obj.layers)) return obj.layers;
    for (const key in obj) {
        const val = obj[key];
        if (Array.isArray(val) && val.length > 0) {
            if (val[0] && (val[0].type || val[0].box)) {
                return val;
            }
        }
    }
    return [];
};

const parseCoord = (val: any, baseSize: number = 100): number => {
    if (typeof val === 'number') {
        // Keep decimals for precision (don't Math.round too early)
        return val;
    }
    if (typeof val === 'string') {
        return parseFloat(val.replace('%', ''));
    }
    return 0;
};

const normalizeElement = (el: any, index: number) => {
    let box = el.box;

    if (!box) {
        box = { top: 10, left: 10, width: 20, height: 20 };
    } else if (Array.isArray(box)) {
        if (box.length >= 4) {
            box = { top: box[0], left: box[1], width: box[2], height: box[3] };
        } else {
            box = { top: 0, left: 0, width: 10, height: 10 };
        }
    }

    const nBox = {
        top: parseCoord(box.top),
        left: parseCoord(box.left),
        width: parseCoord(box.width),
        height: parseCoord(box.height)
    };

    // Heuristic: If all values are <= 1, assume they are normalized (0-1) and convert to percentages (0-100).
    // If any value > 1, assume they are already percentages or pixels relative to 100x100 space.
    if (nBox.top <= 1 && nBox.left <= 1 && nBox.width <= 1 && nBox.height <= 1 && (nBox.width > 0 || nBox.height > 0)) {
        nBox.top *= 100;
        nBox.left *= 100;
        nBox.width *= 100;
        nBox.height *= 100;
    }

    let normalizedType = ElementType.VISUAL;
    if (el.type) {
        const upperType = el.type.toUpperCase();
        if (upperType.includes('TEXT') || upperType.includes('TXT')) {
            normalizedType = ElementType.TEXT;
        } else {
            normalizedType = ElementType.VISUAL;
        }
    }

    return {
        ...el,
        id: el.id || `el-${Date.now()}-${index}`,
        type: normalizedType,
        box: nBox,
        originalBox: { ...nBox },
        style: normalizedType === ElementType.TEXT ? (el.style || { fontSize: 'medium', fontWeight: 'normal', color: '#000000', alignment: 'left' }) : undefined,
        content: el.content || (normalizedType === ElementType.TEXT ? "Detected Text" : ""),
        description: el.description || (normalizedType === ElementType.VISUAL ? "Visual Element" : "")
    };
};

const callGeminiWithRetry = async <T>(
    fn: () => Promise<T>,
    retries = 3,
    delay = 2000
): Promise<T> => {
    try {
        return await fn();
    } catch (error: any) {
        const isRateLimit =
            error?.status === 429 ||
            error?.response?.status === 429 ||
            (typeof error?.message === 'string' && (
                error.message.includes('429') ||
                error.message.includes('RESOURCE_EXHAUSTED') ||
                error.message.includes('quota')
            )) ||
            (error?.error?.code === 429);

        if (isRateLimit && retries > 0) {
            console.warn(`Quota/Rate limit hit. Retrying in ${delay}ms... (${retries} remaining)`);
            await new Promise(resolve => setTimeout(resolve, delay));
            return callGeminiWithRetry(fn, retries - 1, delay * 2);
        }
        throw error;
    }
};

const getOpenAIHeaders = (apiKey: string) => ({
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
});

const getOpenAIBaseUrl = (baseUrl: string) => baseUrl.replace(/\/$/, '');

const callOpenAIChat = async (
    systemPrompt: string,
    userPrompt: string,
    imageBase64?: string,
    jsonMode: boolean = false
): Promise<string> => {
    const config = currentSettings.openai;
    const messages: any[] = [
        { role: "system", content: systemPrompt },
        {
            role: "user",
            content: imageBase64
                ? [
                    { type: "text", text: userPrompt },
                    { type: "image_url", image_url: { url: `data:image/png;base64,${imageBase64}` } }
                ]
                : userPrompt
        }
    ];

    const body: any = {
        model: config.recognitionModel,
        messages: messages,
        max_tokens: 16384, // Increased to ensure JSON isn't cut off
    };

    if (jsonMode) {
        body.response_format = { type: "json_object" };
    }

    const res = await fetch(`${getOpenAIBaseUrl(config.baseUrl)}/chat/completions`, {
        method: 'POST',
        headers: getOpenAIHeaders(config.apiKey),
        body: JSON.stringify(body)
    });

    if (!res.ok) {
        const err = await res.text();
        throw new Error(`OpenAI Chat Error (${res.status}): ${err}`);
    }

    const data = await res.json();
    return data.choices?.[0]?.message?.content || "";
};

const callOpenAIImageGen = async (prompt: string, inputImageBase64?: string): Promise<string> => {
    const config = currentSettings.openai;
    const isGeminiModel = config.drawingModel.toLowerCase().includes('gemini');

    if (isGeminiModel) {
        const baseUrl = getOpenAIBaseUrl(config.baseUrl);
        const headers = getOpenAIHeaders(config.apiKey);

        const messages = [
            {
                role: 'user',
                content: inputImageBase64
                    ? [
                        { type: "text", text: prompt },
                        { type: "image_url", image_url: { url: `data:image/png;base64,${inputImageBase64}` } }
                    ]
                    : prompt
            }
        ];

        const res = await fetch(`${baseUrl}/chat/completions`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                model: config.drawingModel,
                messages: messages
            })
        });

        if (!res.ok) {
            const err = await res.text();
            throw new Error(`Gemini-via-OpenAI Image Gen Error (${res.status}): ${err}`);
        }

        const data = await res.json();
        const content = data.choices?.[0]?.message?.content || "";

        const match = content.match(/\!\[.*?\]\((.*?)\)/);
        let imageUrl = "";

        if (match && match[1]) {
            imageUrl = match[1];
        } else if (content.trim().startsWith('http')) {
            imageUrl = content.trim();
        }

        if (imageUrl) {
            if (imageUrl.startsWith('data:image')) {
                return imageUrl.split(',')[1];
            }
            try {
                const imgRes = await fetch(imageUrl);
                const blob = await imgRes.blob();
                const b64 = await blobToBase64(blob);
                return b64.split(',')[1];
            } catch (e) {
                console.warn("Failed to fetch image from proxy URL:", e);
            }
        }
        return "";
    }

    const isDalle3 = config.drawingModel.includes('dall-e-3');
    const size = isDalle3 ? "1792x1024" : "1024x1024";

    const res = await fetch(`${getOpenAIBaseUrl(config.baseUrl)}/images/generations`, {
        method: 'POST',
        headers: getOpenAIHeaders(config.apiKey),
        body: JSON.stringify({
            model: config.drawingModel,
            prompt: prompt,
            n: 1,
            size: size,
            response_format: "b64_json"
        })
    });

    if (!res.ok) {
        const err = await res.text();
        throw new Error(`OpenAI Image Gen Error (${res.status}): ${err}`);
    }

    const data = await res.json();
    return data.data?.[0]?.b64_json || "";
};

const getGeminiClient = (overrideConfig?: ProviderConfig) => {
    const config = overrideConfig || currentSettings.gemini;
    if (!config.apiKey) throw new Error("Gemini API Key is missing.");
    const options: any = {
        apiKey: config.apiKey,
        httpOptions: {
            baseUrl: config.baseUrl
        }
    };
    return new GoogleGenAI(options);
};

export const testModel = async (
    type: 'recognition' | 'drawing',
    provider: 'gemini' | 'openai',
    config: ProviderConfig
): Promise<{ success: boolean; message: string }> => {
    try {
        if (!config.apiKey) return { success: false, message: "API Key Missing" };

        if (provider === 'gemini') {
            const ai = getGeminiClient(config);
            const modelName = type === 'recognition' ? config.recognitionModel : config.drawingModel;
            if (!modelName) return { success: false, message: "Model Name Missing" };
            try {
                await ai.models.generateContent({
                    model: modelName,
                    contents: { parts: [{ text: "Hello" }] }
                });
                return { success: true, message: "Connected" };
            } catch (e: any) {
                const errMsg = e.message || "";
                if (errMsg.includes('404') || errMsg.includes('not found')) throw new Error(`Model '${modelName}' not found`);
                if (errMsg.includes('403') || errMsg.includes('PERMISSION_DENIED')) return { success: false, message: "403 Forbidden" };
                if (errMsg.includes('400') || errMsg.includes('INVALID_ARGUMENT')) return { success: true, message: "Connected (Arg Error OK)" };
                throw e;
            }
        } else if (provider === 'openai') {
            const baseUrl = getOpenAIBaseUrl(config.baseUrl);
            const headers = getOpenAIHeaders(config.apiKey);
            if (type === 'recognition') {
                const res = await fetch(`${baseUrl}/chat/completions`, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({
                        model: config.recognitionModel,
                        messages: [{ role: 'user', content: 'Ping' }],
                        max_tokens: 5
                    })
                });
                if (!res.ok) throw new Error(`${res.status}`);
                return { success: true, message: "Connected" };
            } else {
                const isGemini = config.drawingModel.toLowerCase().includes('gemini');
                if (isGemini) {
                    const res = await fetch(`${baseUrl}/chat/completions`, {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify({ model: config.drawingModel, messages: [{ role: 'user', content: 'Test Image Gen' }] })
                    });
                    if (!res.ok) throw new Error(`${res.status}`);
                    return { success: true, message: "Connected (Proxy)" };
                } else {
                    const res = await fetch(`${baseUrl}/images/generations`, {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify({ model: config.drawingModel, prompt: "A red circle", n: 1, size: "1024x1024" })
                    });
                    if (!res.ok) throw new Error(`${res.status}`);
                    return { success: true, message: "Connected" };
                }
            }
        }
        return { success: false, message: "Unknown Provider" };
    } catch (error: any) {
        return { success: false, message: error.message || "Unknown error" };
    }
};

/**
 * 2. Background Generation with Explicit Text Removal
 * Uses detected text and their bounding boxes to inform the model what to remove.
 */
export const removeTextFromImage = async (base64Image: string, detectedElements: SlideTextElement[]): Promise<string | null> => {
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

    // STRICTER PROMPT FOR SURGICAL REMOVAL
    let prompt = "Strictly preserve the original aspect ratio (16:9). You are an expert image editor performing surgical text removal.";

    if (detectedElements && detectedElements.length > 0) {
        prompt += " \n\nCRITICAL INSTRUCTIONS:\n1. I have identified specific text regions below.\n2. You MUST erase content ONLY within these exact bounding boxes (coordinates 0-100%).\n3. DO NOT touch, blur, remove, or alter any other part of the image. Visual elements, icons, lines, and shapes outside these boxes must remain PIXEL-PERFECT identical to the original.\n";

        detectedElements.forEach((el, index) => {
            const { top, left, width, height } = el.box;
            if (index < 30) {
                prompt += ` - Target Zone ${index + 1}: [Top: ${top.toFixed(2)}%, Left: ${left.toFixed(2)}%, Width: ${width.toFixed(2)}%, Height: ${height.toFixed(2)}%] (Content to erase: "${el.content.substring(0, 20)}...")\n`;
            }
        });
        prompt += "\nFill these erased text zones with the matching background color or texture of the immediate surrounding area. Do not hallucinate new objects.";
    }

    prompt += " \n\nOutput ONLY the processed image.";

    try {
        if (currentSettings.currentProvider === 'gemini') {
            const ai = getGeminiClient();
            const response = await callGeminiWithRetry<GenerateContentResponse>(() => ai.models.generateContent({
                model: currentSettings.gemini.drawingModel,
                contents: {
                    parts: [
                        { inlineData: { mimeType: 'image/png', data: cleanBase64 } },
                        { text: prompt }
                    ]
                },
                // Add safety settings to prevent false positives blocking image generation
                safetySettings: [
                    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                ]
            }));
            if (response.candidates?.[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if (part.inlineData && part.inlineData.data) {
                        return part.inlineData.data;
                    }
                }
            }
        } else {
            const config = currentSettings.openai;
            const isGeminiDrawing = config.drawingModel.toLowerCase().includes('gemini');

            if (isGeminiDrawing) {
                return await callOpenAIImageGen(prompt, cleanBase64);
            } else {
                const description = await callOpenAIChat(
                    "Graphic Designer",
                    "Detailed analysis of this PPT slide's background style, color palette, and layout structure. IGNORE the text content. Describe ONLY the visual design elements (background shapes, gradients, footer style, icon placement) so a designer can recreate the empty template.",
                    cleanBase64
                );

                if (description) {
                    return await callOpenAIImageGen(`Professional PowerPoint Background, 16:9 Wide Aspect Ratio. Exact replica of this style: ${description}. NO TEXT. Clean, empty template for presentation.`);
                }
            }
        }
        return null;
    } catch (error) {
        console.warn("Background gen failed silently:", error);
        return null;
    }
};

/**
 * 2.1 Explicit Erasure of User Defined Regions (Partial Inpainting)
 */
export const eraseAreasFromImage = async (base64Image: string, boxes: BoundingBox[]): Promise<string | null> => {
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

    let prompt = "Strictly preserve the original aspect ratio (16:9). You are an expert image editor performing precise object removal.";

    if (boxes && boxes.length > 0) {
        prompt += " \n\nCRITICAL INSTRUCTIONS:\n1. I have identified specific regions below that contain unwanted elements.\n2. You MUST erase EVERYTHING inside these exact bounding boxes (coordinates 0-100%).\n3. DO NOT touch, blur, remove, or alter any other part of the image outside these boxes. The rest of the image must remain PIXEL-PERFECT.\n";

        boxes.forEach((box, index) => {
            const { top, left, width, height } = box;
            prompt += ` - Removal Zone ${index + 1}: [Top: ${top.toFixed(2)}%, Left: ${left.toFixed(2)}%, Width: ${width.toFixed(2)}%, Height: ${height.toFixed(2)}%]\n`;
        });
        prompt += "\nFill these erased zones by extending the surrounding background texture, color, or pattern naturally. Make it look like nothing was ever there.";
    } else {
        return null;
    }

    prompt += " \n\nOutput ONLY the processed image.";

    try {
        if (currentSettings.currentProvider === 'gemini') {
            const ai = getGeminiClient();
            const response = await callGeminiWithRetry<GenerateContentResponse>(() => ai.models.generateContent({
                model: currentSettings.gemini.drawingModel,
                contents: {
                    parts: [
                        { inlineData: { mimeType: 'image/png', data: cleanBase64 } },
                        { text: prompt }
                    ]
                },
                safetySettings: [
                    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                ]
            }));
            if (response.candidates?.[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if (part.inlineData && part.inlineData.data) {
                        return part.inlineData.data;
                    }
                }
            }
        } else {
            const config = currentSettings.openai;
            const isGeminiDrawing = config.drawingModel.toLowerCase().includes('gemini');
            if (isGeminiDrawing) {
                return await callOpenAIImageGen(prompt, cleanBase64);
            } else {
                const description = await callOpenAIChat("Assistant", "Describe this image background structure ignoring specific objects.", cleanBase64);
                return await callOpenAIImageGen(`Image: ${description}. Remove objects in specified areas and fill with background.`, cleanBase64);
            }
        }
        return null;
    } catch (error) {
        console.error("Erasure failed:", error);
        throw error;
    }
};

export const regenerateVisualElement = async (croppedBase64: string, instruction: string): Promise<string | null> => {
    const cleanBase64 = croppedBase64.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");
    try {
        if (currentSettings.currentProvider === 'gemini') {
            const ai = getGeminiClient();
            const response = await callGeminiWithRetry<GenerateContentResponse>(() => ai.models.generateContent({
                model: currentSettings.gemini.drawingModel,
                contents: {
                    parts: [
                        { inlineData: { mimeType: 'image/png', data: cleanBase64 } },
                        { text: `Edit this image: ${instruction}. Maintain exact aspect ratio and style. Output image only.` }
                    ]
                },
                safetySettings: [
                    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                ]
            }));
            if (response.candidates?.[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if (part.inlineData && part.inlineData.data) {
                        return part.inlineData.data;
                    }
                }
            }
        } else {
            const config = currentSettings.openai;
            const isGeminiDrawing = config.drawingModel.toLowerCase().includes('gemini');
            if (isGeminiDrawing) {
                return await callOpenAIImageGen(`Edit this image: ${instruction}. Maintain exact aspect ratio and style. Output image only.`, cleanBase64);
            } else {
                const description = await callOpenAIChat("Assistant", "Describe this image element.", cleanBase64);
                return await callOpenAIImageGen(`Image: ${description}. Modification: ${instruction}. White background.`);
            }
        }
        return null;
    } catch (error) {
        console.error("Regenerate Visual Failed:", error);
        throw error;
    }
}

export const refineElement = async (croppedBase64: string, instruction?: string): Promise<any[]> => {
    const cleanBase64 = croppedBase64.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

    // HARDENED PROMPT FOR PRECISE SUB-ELEMENT DETECTION
    const prompt = `
    STRICT GEOMETRY MODE.
    This input image is a specific cropped region of a slide.
    ${instruction ? `Instruction: ${instruction}` : ''}
    
    Tasks:
    1. Identify distinct sub-elements (text lines, icons, shapes) within this crop.
    2. Define precise bounding boxes RELATIVE TO THIS CROPPED IMAGE (0-100%).
    
    CRITICAL RULES:
    - Bounding boxes must be TIGHT. Do not include empty whitespace or padding.
    - If there is a single centered object, the box should hug its visible pixels exactly.
    - Use decimal percentages (e.g. 12.5) for precision if needed.
    
    Return JSON: { "elements": [{ "type", "content"/"description", "box": { "top": number, "left": number, "width": number, "height": number } }] }
  `;

    try {
        let jsonText = "";
        if (currentSettings.currentProvider === 'gemini') {
            const ai = getGeminiClient();
            const response = await callGeminiWithRetry<GenerateContentResponse>(() => ai.models.generateContent({
                model: currentSettings.gemini.recognitionModel,
                contents: { parts: [{ inlineData: { mimeType: 'image/png', data: cleanBase64 } }, { text: prompt }] },
                config: { responseMimeType: "application/json" }
            }));
            jsonText = response.text || "";
        } else {
            jsonText = await callOpenAIChat("JSON Generator", prompt, cleanBase64, true);
        }
        const data = tryParseJSON(cleanJsonString(jsonText));
        return (findElementsArray(data) || []).map(normalizeElement);
    } catch (error) {
        console.error("Refine Element Failed:", error);
        throw error;
    }
};

const processImageSchema: Schema = {
    type: Type.OBJECT,
    properties: {
        backgroundColor: { type: Type.STRING },
        elements: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    type: { type: Type.STRING, enum: [ElementType.TEXT, ElementType.VISUAL] },
                    content: { type: Type.STRING },
                    description: { type: Type.STRING },
                    box: {
                        type: Type.OBJECT,
                        properties: {
                            top: { type: Type.NUMBER },
                            left: { type: Type.NUMBER },
                            width: { type: Type.NUMBER },
                            height: { type: Type.NUMBER },
                        },
                        required: ["top", "left", "width", "height"],
                    },
                    style: {
                        type: Type.OBJECT,
                        properties: {
                            fontSize: { type: Type.STRING, enum: ["small", "medium", "large", "title"] },
                            fontWeight: { type: Type.STRING, enum: ["normal", "bold"] },
                            color: { type: Type.STRING },
                            alignment: { type: Type.STRING, enum: ["left", "center", "right"] }
                        }
                    }
                },
                required: ["type", "box"],
            },
        },
    },
    required: ["elements", "backgroundColor"],
};

// --- NEW FUNCTION: STEP 1 - LAYOUT ANALYSIS ---
export const analyzeLayout = async (base64Image: string): Promise<SlideAnalysisResult> => {
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

    const prompt = `
    Analyze the layout of this PPT slide using STRICT GEOMETRY.
    
    1. **Background**: Identify the dominant solid background color.
    2. **Elements Detection**: Identify ALL bounding boxes.
    
    IMPORTANT for Bounding Boxes:
    - Draw the TIGHTEST possible bounding box around the actual visible pixels of the element.
    - EXCLUDE whitespace, transparent padding, or empty areas around icons/text.
    - If an element is at coordinates [10, 10], do NOT say [0, 0] just to be safe. Be precise.
    - Use decimal values (e.g. 15.5) for higher precision.
    - Return coordinates in 0-100% relative to the image size.
  `;

    let jsonText = "";
    try {
        if (currentSettings.currentProvider === 'gemini') {
            const ai = getGeminiClient();
            const analysisResponse = await callGeminiWithRetry<GenerateContentResponse>(() => ai.models.generateContent({
                model: currentSettings.gemini.recognitionModel,
                contents: { parts: [{ inlineData: { mimeType: 'image/png', data: cleanBase64 } }, { text: prompt + " Strictly follow the JSON schema." }] },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: processImageSchema
                }
            }));
            jsonText = analysisResponse.text || "";
        } else {
            // OpenAI Prompt - Explicit JSON Example instead of Schema Object
            const openAIPrompt = `${prompt}
        
        Strictly output VALID JSON with this structure (no markdown code blocks):
        {
          "backgroundColor": "#ffffff",
          "elements": [
            {
              "type": "TEXT", // or "VISUAL"
              "content": "string (for text)",
              "description": "string (for visual)",
              "box": { "top": 10.5, "left": 10.2, "width": 50.0, "height": 20.1 },
              "style": {
                "fontSize": "medium", // small, medium, large, title
                "fontWeight": "normal", // normal, bold
                "color": "#000000",
                "alignment": "left"
              }
            }
          ]
        }`;

            jsonText = await callOpenAIChat("JSON Generator", openAIPrompt, cleanBase64, true);
        }

        const cleanedJson = cleanJsonString(jsonText);
        const analysisData = tryParseJSON(cleanedJson);
        const rawElements = findElementsArray(analysisData);
        const processedElements = rawElements.map(normalizeElement);

        return {
            backgroundColor: analysisData.backgroundColor || '#ffffff',
            elements: processedElements,
            cleanedImage: null, // No inpainting yet
            rawResponse: jsonText
        };

    } catch (error: any) {
        console.error("Layout Analysis Failed:", error);
        error.rawResponse = jsonText;
        throw error;
    }
};

// --- NEW FUNCTION: STEP 3 - FINAL PROCESSING ---
export const processConfirmedLayout = async (base64Image: string, confirmedElements: any[], backgroundColor: string): Promise<SlideAnalysisResult> => {
    // 1. Filter Text elements
    const detectedTextElements = confirmedElements.filter(el => el.type === ElementType.TEXT) as SlideTextElement[];

    // 2. Background Generation using the confirmed boxes as mask guidance
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

    try {
        const cleanedImageBase64 = await removeTextFromImage(cleanBase64, detectedTextElements);

        // Critical Fix: If cleanedImageBase64 is null (failed), fallback to original base64Image
        // This prevents the "background gone" issue.
        const finalImage = cleanedImageBase64 ? `data:image/png;base64,${cleanedImageBase64}` : base64Image;

        if (!cleanedImageBase64) {
            console.warn("Background text removal failed or returned empty. Reverting to original image.");
        }

        return {
            backgroundColor: backgroundColor,
            elements: confirmedElements,
            cleanedImage: finalImage,
            rawResponse: cleanedImageBase64 ? "Success" : "Background Gen Failed (Fallback to Original)"
        };
    } catch (e) {
        console.error("Final Processing Failed:", e);
        // Fallback on error to ensure user still sees the slide, albeit with original text
        return {
            backgroundColor: backgroundColor,
            elements: confirmedElements,
            cleanedImage: base64Image,
            rawResponse: "Background Gen Error (Fallback to Original)"
        };
    }
};

// --- LEGACY/WRAPPER FUNCTION (For backward compatibility if needed) ---
export const processImage = async (base64Image: string): Promise<SlideAnalysisResult> => {
    const analysis = await analyzeLayout(base64Image);
    return await processConfirmedLayout(base64Image, analysis.elements, analysis.backgroundColor);
};

export const analyzeVisualToVector = async (base64Image: string): Promise<{
    isVector: boolean;
    shapeType: PPTShapeType;
    style: {
        fillColor: string;
        strokeColor: string;
        strokeWidth: number;
        opacity: number;
    }
}> => {
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

    const prompt = `
    Analyze this cropped image content strictly within its boundaries. 
    
    **STRICT COORDINATE RULE:**
    You are analyzing a specific visual element that the user has manually cropped. 
    DO NOT infer shapes or lines that exist outside these visible pixels.
    
    Determine if it can be represented as a basic PowerPoint vector AutoShape.
    
    **CRITICAL RULE: isVector = true ONLY if all of the following are true:**
    1. The image is a simple geometric shape (Rectangle, Circle, Triangle, Arrow, Star, Line, Polygon).
    2. It has a FLAT, SOLID fill color (or simple transparency).
    3. It has a SIMPLE stroke/border or no border.
    
    **CRITICAL RULE: isVector = false if ANY of the following are true:**
    1. It is a Photograph, Screenshot, or Realistic Object.
    2. It contains GRADIENTS, Shadows, Textures, or 3D effects.
    3. It is a complex icon or logo with multiple distinct colors/parts that cannot be a single shape.
    4. It contains text embedded inside (unless it's just a simple shape container).
    
    Question 1: Is this a simple vector shape based on the strict rules above?
    - Return isVector: true ONLY if it is a simple shape that can be perfectly recreated with SVG/PPT Shape.
    - Return isVector: false otherwise (it will be treated as an image).

    Question 2: If isVector is true, identify the shape type.
    Options: 'rect', 'roundRect', 'ellipse', 'triangle', 'arrowRight', 'arrowLeft', 'line', 'star', 'pentagon', 'hexagon', 'diamond', 'callout'.
    
    Question 3: Extract style.
    - fillColor (hex)
    - strokeColor (hex)
    - strokeWidth (approx pt)
    - opacity (0-1)
    
    Return JSON: { "isVector": boolean, "shapeType": "...", "style": { ... } }
    `;

    try {
        let jsonText = "";
        if (currentSettings.currentProvider === 'gemini') {
            const ai = getGeminiClient();
            const response = await callGeminiWithRetry<GenerateContentResponse>(() => ai.models.generateContent({
                model: currentSettings.gemini.recognitionModel,
                contents: { parts: [{ inlineData: { mimeType: 'image/png', data: cleanBase64 } }, { text: prompt }] },
                config: { responseMimeType: "application/json" }
            }));
            jsonText = response.text || "";
        } else {
            jsonText = await callOpenAIChat("Shape Analyzer", prompt, cleanBase64, true);
        }

        const data = tryParseJSON(cleanJsonString(jsonText));

        return {
            isVector: data.isVector === true,
            shapeType: data.shapeType || 'rect',
            style: {
                fillColor: data.style?.fillColor || '#cccccc',
                strokeColor: data.style?.strokeColor || 'transparent',
                strokeWidth: data.style?.strokeWidth || 0,
                opacity: data.style?.opacity ?? 1
            }
        };

    } catch (e) {
        console.warn("Vector analysis failed, defaulting to image mode:", e);
        // Fallback to Image mode on failure
        return {
            isVector: false,
            shapeType: 'rect',
            style: { fillColor: '#000000', strokeColor: 'transparent', strokeWidth: 0, opacity: 1 }
        };
    }
};
