/**
 * Report 引用预处理组合式函数
 *
 * 为 Report.vue 提供流式增量预处理能力：
 * - content 单调追加时，仅处理新增 delta，复用历史结果
 * - content 非追加（改写、回退）时，全量重算
 * - docs 稳定，无需做变更检测
 */

import {
  appendReferencesHtml,
  createCitationReferenceState,
  type CitationDoc,
  type CitationReferenceState,
} from "./citationReferenceBuilder";
import { CitationResultCache } from "./citationCache";
import {
  replaceCitationsInSegment,
  type CitationDocs,
} from "./citationPipeline";

type CitationPreprocessorOptions = {
  /** 当前 Report 对应的 docs（由外部透传，可能异步更新） */
  getDocs: () => CitationDocs;
  /** 下载参考文献链接所需的知识库 ID */
  getKnowledgeBaseId: () => string;
  /** 用于 console 分组日志前缀 */
  getReportLogPrefix: () => string;
};

type PreprocessorState = {
  /** 最近一次处理过的原始全文（用于判断是否 append-only） */
  lastRawContent: string;
  /** 最近一次处理后的正文（不含/含 references 由 shouldAddReferences 决定） */
  lastPreprocessedBody: string;
  /** 引用序号状态，跨增量片段复用 */
  referenceState: CitationReferenceState;
};
/** 匹配完整引用标记，如【12】 */
const COMPLETE_CITATION_RE = /【\d+】/;
/** 全局匹配完整引用标记，用于计数/过滤 */
const COMPLETE_CITATION_GLOBAL_RE = /【\d+】/g;
/** 匹配结尾处未闭合引用，如【 / 【1 / 【123 */
const TAILING_INCOMPLETE_CITATION_RE = /【\d*$/;

/**
 * 创建 Report 引用预处理器
 * @returns process(content, shouldAddReferences) 与 reset()
 */
export function useCitationPreprocessor(options: CitationPreprocessorOptions) {
  const citationCache = new CitationResultCache(1200);
  const state: PreprocessorState = {
    lastRawContent: "",
    lastPreprocessedBody: "",
    referenceState: createCitationReferenceState(),
  };

  /**
   * 处理 content，将【数字】替换为 CITE 块，可选追加参考文献
   * @param shouldAddReferences 是否在文末追加参考文献列表，流式阶段通常为 false
   */
  const process = (
    content: string,
    shouldAddReferences: boolean = true
  ): string => {
    const docs = options.getDocs();
    // 分支A：docs 为空时不做 cite 匹配，统一移除【x】避免展示原始标记
    if (Object.keys(docs).length === 0) {
      resetState();
      if (!content.includes("【")) return content;
      const citeCount = (content.match(COMPLETE_CITATION_GLOBAL_RE) || [])
        .length;
      if (citeCount > 0) {
        console.groupCollapsed(
          `${options.getReportLogPrefix()} 引用预处理 (count=${citeCount})`
        );
        console.log("docs为空，无法展示引用结果");
        console.groupEnd();
      }
      return content.replace(COMPLETE_CITATION_GLOBAL_RE, "");
    }

    // append-only = 新内容以前一版全文为前缀，符合流式单调追加
    const isAppendOnly =
      content.length >= state.lastRawContent.length &&
      content.startsWith(state.lastRawContent);

    // 分支B：存在改写/回退，增量状态不再可靠，切到全量重建
    if (!isAppendOnly) {
      console.warn("存在改写/回退，增量状态不再可靠，切到全量重建");
      rebuildByFullContent(content, docs);
    } else {
      // 分支C：append-only，尽量只处理新增 delta
      const delta = content.slice(state.lastRawContent.length);
      if (delta) {
        if (delta.includes("】")) {
          // 可能出现边界截断：上一轮末尾是【1，本轮补上】。
          // 用 suffix + delta 形成扩展片段，保证【x】在同一 segment 内可被匹配。
          const suffix =
            state.lastRawContent.match(TAILING_INCOMPLETE_CITATION_RE)?.[0] ??
            "";
          const segment = suffix + delta;
          state.lastRawContent = content;
          if (!COMPLETE_CITATION_RE.test(segment)) {
            // segment 没有完整【x】，无需进入引用流水线
            state.lastPreprocessedBody += delta;
          } else {
            // baseOffset 为 segment 在 fullContent 中的起始位置
            const baseOffset =
              state.lastRawContent.length - delta.length - suffix.length;
            const processedSegment = replaceCitationsInSegment(
              segment,
              baseOffset,
              {
                docs,
                fullContent: content,
                cache: citationCache,
                referenceState: state.referenceState,
                getReportLogPrefix: options.getReportLogPrefix,
              }
            );
            const bodyPrefix =
              suffix && state.lastPreprocessedBody.endsWith(suffix)
                ? state.lastPreprocessedBody.slice(0, -suffix.length)
                : state.lastPreprocessedBody;
            state.lastPreprocessedBody = bodyPrefix + processedSegment;
          }
        } else {
          // delta 不含 】 不可能出现新的完整【x】。
          // 仅隐藏尾部未闭合引用（如“...【1”），减少流式闪烁。
          const processedDelta = delta.includes("【")
            ? delta.replace(TAILING_INCOMPLETE_CITATION_RE, "")
            : delta;
          state.lastRawContent = content;
          state.lastPreprocessedBody += processedDelta;
        }
      }
    }

    if (!shouldAddReferences) return state.lastPreprocessedBody;
    return appendReferencesHtml(
      state.lastPreprocessedBody,
      state.referenceState,
      options.getKnowledgeBaseId()
    );
  };

  /** 组件卸载或切换消息时调用，清空缓存与状态 */
  const reset = () => {
    resetState();
    citationCache.clear();
  };

  return {
    process,
    reset,
  };

  /**
   * 根据全文内容重建预处理状态
   */
  function rebuildByFullContent(
    content: string,
    docs: Record<string, CitationDoc>
  ) {
    // 全量重建时重置引用序号，避免沿用旧状态导致 references 顺序错乱
    state.lastRawContent = content;
    state.lastPreprocessedBody = "";
    state.referenceState = createCitationReferenceState();
    if (!COMPLETE_CITATION_RE.test(content)) {
      state.lastPreprocessedBody = content;
      return;
    }
    state.lastPreprocessedBody = replaceCitationsInSegment(content, 0, {
      docs,
      fullContent: content,
      cache: citationCache,
      referenceState: state.referenceState,
      getReportLogPrefix: options.getReportLogPrefix,
    });
  }

  /** 仅重置当前消息态，不清缓存（由 reset 决定是否一并 clear） */
  function resetState() {
    state.lastRawContent = "";
    state.lastPreprocessedBody = "";
    state.referenceState = createCitationReferenceState();
  }
}
