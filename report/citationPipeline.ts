/**
 * Report 引用替换流水线
 *
 * 将 markdown 中的【数字】引用标记替换为 <CITE_START>...<CITE_END> 格式。
 * 流程：数字匹配 -> 文字匹配 -> 低分时全量 docs 重排 -> 构建 cite 内容。
 *
 * 支持增量处理：仅对 segment 内的引用做替换，baseOffset 用于在全文中的正确位置提取 query 句。
 */

import { wrapCiteData } from "../cite-constants";
import {
  type CitationBestMatch,
  extractSentenceContainingCitation,
  findBestCitationExcerpt,
  stripCitationMarkers,
} from "./citationMatch";
import { findBestNumberExcerpt, type NumberBestMatch } from "./numberMatch";
import {
  buildCitationCacheKey,
  CitationResultCache,
} from "./citationCache";
import {
  registerReference,
  type CitationDoc,
  type CitationReferenceState,
} from "./citationReferenceBuilder";

/** 当文字匹配分数低于此值时，触发全量 docs 重排 */
const CITATION_LOW_SCORE_THRESHOLD = 0.5;
/** 全量重排后仍低于此值则隐藏引用 */
const RERANK_HIDE_CITATION_THRESHOLD = 0.3;
const ENABLE_GLOBAL_DOC_RERANK = true;

export type CitationDocs = Record<string, CitationDoc>;

/**
 * 替换流水线的上下文，由 useCitationPreprocessor 注入
 * - segment: 当前要扫描替换的文本片段（全文 or 增量）
 * - fullContent: 当前完整正文，用于提取 querySentence
 * - cache/referenceState: 跨 chunk 复用状态
 */
type CitationContext = {
  docs: CitationDocs;
  fullContent: string;
  cache: CitationResultCache;
  referenceState: CitationReferenceState;
  getReportLogPrefix: () => string;
};

/**
 * 对指定片段内的【数字】引用进行替换
 * @param segment 待处理文本片段（可为全文或增量 delta）
 * @param baseOffset segment 在全文中的起始偏移，用于 extractSentenceContainingCitation
 */
export function replaceCitationsInSegment(
  segment: string,
  baseOffset: number,
  ctx: CitationContext
): string {
  // 仅用于日志展示本轮 segment 内命中的引用个数
  const totalCitations = (segment.match(/【(\d+)】/g) || []).length;
  let logFlag = false;
  const openPreprocessGroup = () => {
    if (logFlag || totalCitations === 0) return;
    logFlag = true;
    console.groupCollapsed(
      `${ctx.getReportLogPrefix()} 引用预处理 (count=${totalCitations})`
    );
  };

  const replaced = segment.replace(/【(\d+)】/g, (match, citeId, offset) => {
    openPreprocessGroup();
    // 将 segment 内 offset 映射回 fullContent 的绝对偏移
    const citeOffset = Number(offset ?? 0) + baseOffset;
    const cleanMatch = String(citeId ?? "").trim();
    console.warn(`原始引用=${cleanMatch} | offset=${citeOffset}`);

    // step 1: 先按 citeId 找到候选 doc；找不到直接删除该引用标记
    const mapData = ctx.docs[cleanMatch];
    if (!mapData) {
      console.error("引用处理失败：未找到对应 docs", {
        citeId: cleanMatch,
        offset: citeOffset,
        内容预览: toPreview(ctx.fullContent),
      });
      return "";
    }

    // step 2: 从 fullContent 提取引用附近查询句（不是从 segment 提取）
    const rawSentence = extractSentenceContainingCitation(
      ctx.fullContent,
      citeOffset,
      match.length
    );
    const querySentence = stripCitationMarkers(rawSentence);
    console.warn(querySentence);

    const cacheKey = buildCitationCacheKey(cleanMatch, querySentence);
    const cached = ctx.cache.get(cacheKey);
    // step 3: querySentence 相同则直接复用，避免重复跑匹配逻辑
    if (cached !== undefined) return cached;

    console.log("1) 输入信息", {
      report: ctx.getReportLogPrefix(),
      引用标记: match,
      citeId: cleanMatch,
      offset: citeOffset,
      命中docs: Boolean(mapData),
    });

    const computed = resolveCitationReplacement({
      citeId: cleanMatch,
      querySentence,
      docs: ctx.docs,
      mapData,
      referenceState: ctx.referenceState,
    });

    ctx.cache.set(cacheKey, computed);
    return computed;
  });

  if (logFlag) console.groupEnd();
  return replaced;
}

/**
 * 解析单条引用：数字匹配 -> 文字匹配 -> 可选全量重排 -> 构建 cite 块
 */
function resolveCitationReplacement(args: {
  citeId: string;
  querySentence: string;
  docs: CitationDocs;
  mapData: CitationDoc;
  referenceState: CitationReferenceState;
}): string {
  const { citeId, querySentence, docs, mapData, referenceState } = args;
  const source = buildSearchSource(mapData);

  let best: NumberBestMatch | CitationBestMatch | null = null;
  let finalDocId = citeId;
  let finalDoc = mapData;
  let finalSource = source;
  let currentTextScore: number | null = null;
  let matchStage = "none";

  // 阶段1：数字匹配（命中时优先采用，跳过文字匹配）
  best = findBestNumberExcerpt(source, querySentence, {
    maxSpan: 5,
    maxExcerptLength: 500,
  });
  if (best) {
    matchStage = "number";
    console.log("2) 数字匹配命中", {
      匹配数字: (best as NumberBestMatch).matchNumber,
      匹配分数: best.score,
    });
  }

  // 阶段2：数字未命中时，走文字相似度匹配
  if (!best) {
    best = findBestCitationExcerpt(source, querySentence, {
      maxSpan: 5,
      maxExcerptLength: 500,
    });
    currentTextScore = best?.score ?? 0;
    matchStage = "text";
  }

  const shouldRerankByAllDocs =
    ENABLE_GLOBAL_DOC_RERANK &&
    currentTextScore !== null &&
    currentTextScore < CITATION_LOW_SCORE_THRESHOLD;

  console.log("3) 正文查询句信息", {
    文本长度: querySentence.length,
    查询句预览: toPreview(querySentence),
  });
  console.log("4) 主流程引用文档匹配结果", {
    匹配阶段:
      matchStage === "number"
        ? "数字匹配"
        : matchStage === "text"
          ? "文字匹配"
          : "未命中",
    主流程分数: best?.score ?? 0,
    命中片段预览: toPreview(best?.text),
  });
  console.log("5) 重排触发判断", {
    开关开启: ENABLE_GLOBAL_DOC_RERANK,
    当前文字分数: currentTextScore ?? "N/A(数字匹配直接命中)",
    触发阈值: CITATION_LOW_SCORE_THRESHOLD,
    是否触发全量docs重排: shouldRerankByAllDocs,
  });

  if (shouldRerankByAllDocs) {
    // 阶段3：低分时在全量 docs 上重排，允许从其它 doc 抢占最佳匹配
    type RerankBest = {
      docId: string;
      doc: CitationDoc;
      source: string;
      match: CitationBestMatch | null;
    };

    const rerankSummary: Array<{
      候选docId: string;
      候选文档: string;
      匹配分数: number;
      子句跨度: number;
      是否命中: boolean;
    }> = [];
    let rerankBest: RerankBest | null = null;

    for (const [docId, doc] of Object.entries(docs)) {
      const docSource = buildSearchSource(doc);
      const docMatch = findBestCitationExcerpt(docSource, querySentence, {
        maxSpan: 5,
        maxExcerptLength: 500,
      });

      rerankSummary.push({
        候选docId: docId,
        候选文档: doc.source.replace(/\.\w{1,4}$/, ""),
        匹配分数: docMatch?.score ?? 0,
        子句跨度: docMatch?.span ?? 0,
        是否命中: Boolean(docMatch),
      });

      if (!docMatch) continue;
      if (!rerankBest || docMatch.score > (rerankBest.match?.score ?? 0)) {
        rerankBest = {
          docId,
          doc,
          source: docSource,
          match: docMatch,
        };
      }
    }

    console.log("6) 全量文档相似度匹配候选");
    console.table(rerankSummary);

    if (rerankBest?.match) {
      best = rerankBest.match;
      finalDocId = rerankBest.docId;
      finalDoc = rerankBest.doc;
      finalSource = rerankBest.source;
      matchStage =
        finalDocId === citeId ? "text-rerank-self" : "text-rerank-other-doc";
    }
  }

  console.log("7) 最终采用结果", {
    原引用标记: citeId,
    最终采用文档: finalDocId,
    最终阶段: matchStage,
    最终分数: best?.score ?? 0,
    最终片段预览: toPreview(best?.text),
  });

  if (shouldRerankByAllDocs && (best?.score ?? 0) < RERANK_HIDE_CITATION_THRESHOLD) {
    // 阶段4：触发过全量重排但仍低分，保守隐藏该引用
    console.error("7.1) 全量重排后分数仍低，隐藏引用标记", {
      原citeId: citeId,
      最终采用docId: finalDocId,
      最终分数: best?.score ?? 0,
      隐藏阈值: RERANK_HIDE_CITATION_THRESHOLD,
      说明: "已触发全量docs重排，但置信度仍低于阈值，返回空字符串",
    });
    return "";
  }

  if (!best) {
    // 阶段5：完全未命中，隐藏该引用
    console.error("8) 未命中，隐藏引用标记", {
      来源docId: finalDocId,
      来源文档: finalDoc.source.replace(/\.\w{1,4}$/, ""),
      来源文本预览: toPreview(finalSource),
      说明: "数字匹配与文字匹配均未命中，返回空字符串",
    });
    return "";
  }

  // 阶段6：构建 cite payload，并注册到 references 状态
  const citeContent = buildCiteContent(finalSource, finalDoc, best.text || "");
  const { displaySource } = registerReference(referenceState, finalDoc);
  return wrapCiteData({
    source: displaySource,
    content: citeContent,
  });
}

/** 构建 tooltip 展示的 cite 内容：不足 200 字时向上下文扩展，并对匹配片段加下划线 */
function buildCiteContent(
  finalSource: string,
  finalDoc: CitationDoc,
  originalText: string
): string {
  let citeContent = originalText;

  if (citeContent && citeContent.length < 200) {
    const fullContent = finalSource.replace(/\n/g, "");
    const bestIndex = fullContent.indexOf(originalText);
    if (bestIndex !== -1) {
      const needChars = 200 - citeContent.length;
      const halfNeed = Math.ceil(needChars / 2);
      const start = Math.max(0, bestIndex - halfNeed);
      const end = Math.min(fullContent.length, bestIndex + originalText.length + halfNeed);
      const extendedContent = fullContent.slice(start, end);
      const prefix = start > 0 ? "..." : "";
      const suffix = end < fullContent.length ? "..." : "";
      citeContent = prefix + extendedContent + suffix;
    }
  }

  if (!citeContent) {
    citeContent = finalDoc.content.slice(0, 200) + "...";
  }

  if (originalText && citeContent.includes(originalText)) {
    citeContent = citeContent.replace(
      originalText,
      `<span style="border-bottom: 1px solid #9a9a9a;">${originalText}</span>`
    );
  }
  return citeContent;
}

/** 从 doc 中提取可搜索正文（去掉 source 标题行） */
export function buildSearchSource(doc: CitationDoc): string {
  return (doc.content || "").replace((doc.source || "").trim(), "").trim();
}

function toPreview(text?: string, max = 120): string {
  const raw = (text || "").replace(/\s+/g, " ").trim();
  if (!raw) return "";
  return raw.length > max ? `${raw.slice(0, max)}...` : raw;
}
