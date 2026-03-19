<template>
  <div ref="reportRootRef" class="markdown-report-instance">
    <MarkdownRenderer
      :renderedMarkdown="renderedMarkdown"
      ref="markdownContainer"
      @click="onClick"
      @mouseover="onMouseover"
      @mouseout="onMouseout"
    />

    <el-tooltip
      ref="citeTooltipRef"
      :trigger="[]"
      effect="dark"
      :visible="tooltipVisible"
      :virtual-triggering="true"
      :virtual-ref="virtualRef as any"
      teleported
      append-to="body"
      placement="bottom"
      popper-class="cite-tooltip"
    >
      <template #content>
        <div class="tooltip-content">
          <div class="tooltip-header">
            <div class="tooltip-title">{{ tooltipSource }}</div>
            <div class="tooltip-note">单击上标固定</div>
          </div>
          <div class="tooltip-body" v-html="tooltipContentHtml"></div>
        </div>
      </template>
      <span></span>
    </el-tooltip>
  </div>
</template>

<script setup lang="ts">
import { marked } from "./cite-marked-plugins";
import MarkdownRenderer from "./index.vue";
import { nextTick } from "vue";
import type { ElTooltip } from "element-plus";
import { CITE_END, CITE_START } from "./cite-constants";
import { useCitationPreprocessor } from "./report/useCitationPreprocessor";
import { useMultiChatStore } from "@/stores/multiChat";
import { useUIStore } from "@/stores/ui";

const props = withDefaults(
  defineProps<{
    content: string;
    docs?: Record<string, { index: string; source: string; content: string }>;
    knowledgeBaseId?: string;
    /** 是否追加参考文献列表，仅在正文响应完成时传 true */
    addReferences?: boolean;
    /** 渲染调度延迟(ms)，用于多 Report 时优先渲染底部，0 表示最优先 */
    scheduleDelay?: number;
  }>(),
  { addReferences: true, scheduleDelay: 0 }
);

const chatStore = useMultiChatStore();
const uiStore = useUIStore();
const { isStreamingResponse } = storeToRefs(chatStore);
const reportRootRef = ref<HTMLElement | undefined>();
const loading = ref(true);

/** 【\d+】引用标记的正则，用于 loading 时快速预览 */
const CITE_MARKER_RE = /【\d+】/g;

/** 预览用 HTML 转义 */
function escapePreview(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * loading 时生成快速预览：移除引用标记 + marked 渲染 + 参考文献列表（样式一致，无真实链接）
 */
function getQuickPreviewHtml(raw: string): string {
  const content = raw || "";
  const stripped = content.replace(CITE_MARKER_RE, "");

  let contentToPreview = stripped;
  if (props.docs && Object.keys(props.docs).length > 0) {
    const seen = new Set<string>();
    const refItems: string[] = [];
    let index = 1;
    for (const m of content.match(/【(\d+)】/g) || []) {
      const id = m.replace(/【|】/g, "");
      if (!seen.has(id)) {
        seen.add(id);
        const doc = props.docs[id];
        if (doc) {
          const displaySource = doc.source.replace(/\.\w{1,4}$/, "");
          refItems.push(
            `<div class="ref-item">[${index}] <span class="ref-link">${escapePreview(displaySource)}</span></div>`
          );
          index++;
        }
      }
    }
    if (refItems.length > 0) {
      const refList = `<div class="ref-list">${refItems.join("")}</div>`;
      contentToPreview = `${stripped}\n\n### 参考文献\n\n${refList}`;
    }
  }

  return marked(contentToPreview) as string;
}

function getReportLogPrefix(): string {
  const sessionTitleFromHeader = (
    document.querySelector(".chat-main-header") as HTMLElement | null
  )?.innerText
    ?.trim()
    ?.replace(/\s+/g, " ");
  const sessionTitle =
    sessionTitleFromHeader ||
    chatStore.activeConversation?.title?.trim() ||
    "未命名会话";

  const currentRoot = reportRootRef.value || null;
  const allReportNodes = Array.from(
    document.querySelectorAll(".report-content .markdown-report-instance")
  );
  const index = allReportNodes.findIndex((node) => node === currentRoot) + 1;
  const safeIndex = index > 0 ? index : 1;

  return `${sessionTitle} | Report#${safeIndex}`;
}

const renderedMarkdown = ref("");
const tooltipVisible = ref(false);
const tooltipSource = ref("");
const tooltipContent = ref("");
const tooltipContentHtml = ref("");
const virtualRef = ref<HTMLElement | undefined>();
const citeTooltipRef = ref<InstanceType<typeof ElTooltip> | null>(null);
const tooltipPinned = ref(false);
const citationPreprocessor = useCitationPreprocessor({
  getDocs: () => props.docs || {},
  getKnowledgeBaseId: () => props.knowledgeBaseId || "",
  getReportLogPrefix,
});
const NEIGHBOR_CITATIONS_RE = /【\d+】【[\d【】]+】/;
const HAS_CITE_BLOCK_RE = new RegExp(`${CITE_START}[\\s\\S]*?${CITE_END}`);

/** 渲染任务版本号，用于取消过期的异步渲染 */
let renderVersion = 0;

/**
 * 渲染报告内容
 * 先同步设置 quick preview（去【\d+】），再异步执行完整渲染
 */
const renderReportContent = (rawContent: string) => {
  const version = ++renderVersion;
  const content = rawContent || "";

  const run = () => {
    if (version !== renderVersion) return;
    const reorderedContent =
      props.addReferences && NEIGHBOR_CITATIONS_RE.test(content)
        ? reorderNeighborCitations(content)
        : content;
    const preprocessedContent = citationPreprocessor.process(
      reorderedContent,
      props.addReferences
    );
    const deduplicatedContent = HAS_CITE_BLOCK_RE.test(preprocessedContent)
      ? deduplicateNeighborCitations(preprocessedContent)
      : preprocessedContent;
    const html = marked(deduplicatedContent) as string;
    if (version !== renderVersion) return;
    renderedMarkdown.value = html;
    loading.value = false;
  };

  const delay = props.scheduleDelay ?? 0;
  setTimeout(run, delay);
};

/** docs 变化时重置预处理器，避免沿用空 docs 时的缓存结果 */
watch(
  () => props.docs,
  () => {
    citationPreprocessor.reset();
  },
  { deep: true }
);

watch(
  () => [props.content, props.addReferences, props.docs] as const,
  () => {
    // 首次渲染放到 onMounted，避免挂载前取不到 reportRootRef
    if (!reportRootRef.value) return;
    renderReportContent(props.content || "");
  }
);

/**
 * 按照文档出现顺序，重排相邻的引用标记
 * @param content
 * @todo refMap多次构建，需要优化
 */
function reorderNeighborCitations(content: string): string {
  if (!NEIGHBOR_CITATIONS_RE.test(content)) return content;
  const docs = props.docs || {};
  const refMap: Map<string, { source: string; index: number }> = new Map();

  let index = 1;
  content.match(/【(\d+)】/g)?.map((refIndex) => {
    const cleanRefIndex = refIndex.match(/【(\d+)】/)?.[1]!;
    const doc = docs[cleanRefIndex];
    if (doc && !refMap.get(doc.source)) {
      // 所有相同的文档使用同一索引，仅在未出现过时设置
      refMap.set(doc.source, { source: doc.source, index });
      index++;
    }
  });

  return content.replace(/【\d+】【[\d【】]+】/g, (match) => {
    const cleanRefIndexes = match.match(/\d+/g);
    if (!cleanRefIndexes) return match;
    return cleanRefIndexes
      .sort((a, b) => {
        const docA = docs[a];
        const docB = docs[b];
        if (!docA || !docB) return 0;

        const refA = refMap.get(docA.source);
        const refB = refMap.get(docB.source);
        if (!refA || !refB) return 0;

        return refA.index - refB.index;
      })
      .map((ref) => `【${ref}】`)
      .join("");
  });
}

/**
 * 连续 cite 块的正则：匹配 <CITE_START>xxx<CITE_END>+
 * 定界符少见，content 中含 }}、{ 等也不会误匹配
 */
const CONSECUTIVE_CITE_BLOCK_RE = new RegExp(
  `(${CITE_START}[\\s\\S]*?${CITE_END})+`,
  "g"
);

/**
 * 去除连续的重复 cite 引用（按 source 去重）
 * 匹配 <CITE_START>xxx<CITE_END>+ 的连续块，保留每个 source 首次出现的引用
 */
function deduplicateNeighborCitations(content: string): string {
  if (!HAS_CITE_BLOCK_RE.test(content)) return content;
  return content.replace(CONSECUTIVE_CITE_BLOCK_RE, (block) => {
    const seenSources = new Set<string>();
    const keptCites: string[] = [];
    let pos = 0;

    while (pos < block.length) {
      const startIdx = block.indexOf(CITE_START, pos);
      if (startIdx === -1) break;
      const afterStart = startIdx + CITE_START.length;
      const endIdx = block.indexOf(CITE_END, afterStart);
      if (endIdx === -1) break;

      const jsonStr = block.slice(afterStart, endIdx);
      const citeBlock = block.slice(startIdx, endIdx + CITE_END.length);

      try {
        const data = JSON.parse(jsonStr);
        const source = data?.source ?? "";
        if (!seenSources.has(source)) {
          seenSources.add(source);
          keptCites.push(citeBlock);
        }
      } catch {
        keptCites.push(citeBlock);
      }

      pos = endIdx + CITE_END.length;
    }

    return keptCites.join("");
  });
}

function onMouseover(e: MouseEvent) {
  // 流式响应时不显示tooltip
  if (isStreamingResponse.value) return;
  const target = e.target as HTMLElement;
  const sup = target?.closest?.("sup.md-cite") as HTMLElement | null;
  if (!sup) return;
  tooltipSource.value = sup.getAttribute("data-source") || "";
  tooltipContent.value = sup.getAttribute("data-content") || "";
  tooltipContentHtml.value = decodeHtml(tooltipContent.value);
  virtualRef.value = sup;
  tooltipVisible.value = true;
  nextTick(() => {
    // @ts-ignore 类型定义不暴露 updatePopper 时忽略
    citeTooltipRef.value?.updatePopper?.();
  });
}

function onMouseout(e: MouseEvent) {
  const target = e.target as HTMLElement;
  if (!target) return;
  if (target.matches && target.matches("sup.md-cite")) {
    if (tooltipPinned.value) return; // 固定状态下不关闭
    const related = e.relatedTarget as HTMLElement | null;
    if (!related || !related.closest || !related.closest("sup.md-cite")) {
      tooltipVisible.value = false;
    }
  }
}

function onClick(e: MouseEvent) {
  const target = e.target as HTMLElement;
  // 点击参考文献链接
  const refLink = target?.closest?.(".ref-link") as HTMLElement | null;
  if (refLink) {
    e.preventDefault();
    const docName = refLink.getAttribute("data-doc-name") || "";
    const href = refLink.getAttribute("data-href") || "";
    if (docName && href) uiStore.setShowDocView(true, { docName, href });
    return;
  }
  // 点击引用标记
  const sup = target?.closest?.("sup.md-cite") as HTMLElement | null;
  if (!sup) return;
  tooltipPinned.value = true;
  tooltipSource.value = sup.getAttribute("data-source") || "";
  tooltipContent.value = sup.getAttribute("data-content") || "";
  tooltipContentHtml.value = decodeHtml(tooltipContent.value);
  virtualRef.value = sup;
  tooltipVisible.value = true;
  nextTick(() => {
    // @ts-ignore
    citeTooltipRef.value?.updatePopper?.();
  });
}

function onGlobalClick(e: MouseEvent) {
  if (!tooltipVisible.value) return;
  if (!tooltipPinned.value) return;
  const target = e.target as HTMLElement | null;
  if (!target) return;
  const isSup = !!target.closest?.("sup.md-cite");
  const popperEl = document.querySelector(
    ".cite-tooltip"
  ) as HTMLElement | null;
  const isInTooltip = !!(popperEl && popperEl.contains(target));
  if (!isSup && !isInTooltip) {
    tooltipPinned.value = false;
    tooltipVisible.value = false;
    virtualRef.value = undefined;
  }
}

onMounted(() => {
  renderedMarkdown.value = getQuickPreviewHtml(props.content || "");
  renderReportContent(props.content || "");
  window.addEventListener("click", onGlobalClick, true);
});

onBeforeUnmount(() => {
  citationPreprocessor.reset();
  window.removeEventListener("click", onGlobalClick, true);
});

/**
 * 解码 data-* 中的 HTML 实体，允许在 tooltip 中使用 v-html 渲染 <strong> 等标签
 */
function decodeHtml(input: string): string {
  return input
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}
</script>

<style scoped>
.markdown-body {
  font-size: 1rem;

  & :deep() h1,
  & :deep() h2,
  & :deep() h3,
  & :deep() h4,
  & :deep() h5,
  & :deep() h6 {
    border-bottom: none;
  }

  & :deep() hr {
    height: 1px;
  }

  & :deep() sup {
    margin: 0 0.1rem;
    color: var(--icon-blue);
    cursor: pointer;
  }

  & :deep() .ref-list {
    margin: 0.5rem 0;
  }

  & :deep() .ref-item {
    margin-bottom: 0.25rem;
  }

  & :deep() .ref-link {
    color: inherit;
    text-decoration: none;
    cursor: pointer;
    color: #0969da;

    &:hover {
      text-decoration: underline;
    }
  }
}

.tooltip-content {
  display: flex;
  max-width: min(50vw, 30rem);
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.5rem;
}

.tooltip-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tooltip-title {
  font-weight: 600;
}

.tooltip-note {
  color: var(--pale-gray);
  font-size: 0.6rem;
}

.tooltip-body {
  color: var(--pale-gray);

  & :deep() strong {
    padding: 0 0.25rem;
    color: white;
    text-decoration: underline;
  }
}
</style>
