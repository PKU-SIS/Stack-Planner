/**
 * Report 参考文献列表构建
 *
 * 负责维护引用文档的显示顺序、生成参考文献 HTML 区块，
 * 以及文档下载链接的组装。与 cite-marked-plugins 配合使用。
 */

export type CitationDoc = { index: string; source: string; content: string };

/** 参考文献列表中单条目的展示信息 */
export type CitationReferenceItem = {
  index: number;
  displaySource: string;
  docName: string;
};

/** 参考文献构建过程中的状态，跨增量片段共享 */
export type CitationReferenceState = {
  referenceMap: Map<string, CitationReferenceItem>;
  referenceIndex: number;
};

const DOWNLOAD_ORIGIN = "https://stack-planner.pkubir.cn";

/** 创建空的参考文献状态 */
export function createCitationReferenceState(): CitationReferenceState {
  return {
    referenceMap: new Map<string, CitationReferenceItem>(),
    referenceIndex: 1,
  };
}

/**
 * 注册文档到参考文献列表，若已存在则返回现有 displaySource
 * @returns displaySource 去掉扩展名后的文件名，供 cite 块和参考文献链接使用
 */
export function registerReference(
  state: CitationReferenceState,
  doc: CitationDoc
): { displaySource: string } {
  const displaySource = doc.source.replace(/\.\w{1,4}$/, "");
  if (!state.referenceMap.has(displaySource)) {
    state.referenceMap.set(displaySource, {
      index: state.referenceIndex,
      displaySource,
      docName: doc.source,
    });
    state.referenceIndex += 1;
  }
  return { displaySource };
}

/**
 * 在正文末尾追加参考文献 HTML 区块
 * @param content 已预处理后的正文（含 CITE 块）
 * @param state 当前参考文献状态
 * @param kbId 知识库 ID，用于构建下载链接
 */
export function appendReferencesHtml(
  content: string,
  state: CitationReferenceState,
  kbId: string
): string {
  if (state.referenceMap.size === 0) return content;
  const refItems = Array.from(state.referenceMap.values())
    .sort((a, b) => a.index - b.index)
    .map((item) => {
      const url = `${DOWNLOAD_ORIGIN}/download/v1/document_api/download?kb_id=${encodeURIComponent(kbId)}&doc_name=${encodeURIComponent(item.docName)}`;
      return `<div class="ref-item">[${item.index}] <span class="ref-link" data-doc-name="${escapeAttr(item.docName)}" data-href="${escapeAttr(url)}">${escapeAttr(item.displaySource)}</span></div>`;
    })
    .join("");

  const referencesHtml = `<div class="ref-list">${refItems}</div>`;
  return `${content}\n\n### 参考文献\n\n${referencesHtml}`;
}

/** HTML 属性转义，防止 XSS */
function escapeAttr(input: string): string {
  return String(input)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
