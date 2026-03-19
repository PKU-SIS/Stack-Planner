/**
 * Report 引用内容匹配工具（文字相似度）
 *
 * 目标：已知 markdown 中“包含【数字】引用标记的句子”（query），
 * 在引文原文 mapData.content 中找到与 query 最匹配的片段（允许横跨多个子句）。
 *
 * 实现要点：
 * - 先从 markdown content 里根据引用标记的位置提取所在句子（querySentence）
 * - 对 mapData.content 做子句切分，再做 1/2/3 子句拼接候选
 * - 用 bigram Jaccard + coverage 打分，选最高分
 */

export type CitationBestMatch = {
  /** 最佳匹配片段（原文） */
  text: string;
  /** 置信分数（0~1） */
  score: number;
  /** 起始子句 index */
  clauseIndex: number;
  /** 横跨子句数量（1/2/3） */
  span: number;
};

// 句子边界：中文/英文常见句末 + 换行（markdown 里经常用换行分段）
const isBoundary = (ch: string) =>
  ch === "。" ||
  ch === "！" ||
  ch === "？" ||
  ch === "；" ||
  ch === ";" ||
  ch === "!" ||
  ch === "?" ||
  ch === "\n" ||
  ch === "\r";

/**
 * 从 markdown content 中提取“引用标记前的句子片段”
 * @param content - markdown 文本
 * @param citeOffset - `【数字】` 匹配的起始 offset（来自 replace 回调）
 * @param citeLength - `【数字】` 匹配的长度
 */
export function extractSentenceContainingCitation(
  content: string,
  citeOffset: number,
  citeLength: number
): string {
  if (!content) return "";
  const startFrom = Math.max(0, Math.min(citeOffset, content.length));

  /**
   * 修改后的逻辑：
   * - 去除兜底：如果引用标记出现在段首，返回空字符串
   * - 左边界不超过换行符，或先前的引用标记
   */
  const isTrailingWrapper = (ch: string) =>
    ch === "\u201D" || // 右双引号
    ch === "\u2019" || // 右单引号
    ch === '"' ||
    ch === "'" ||
    ch === "）" ||
    ch === ")" ||
    ch === "】" ||
    ch === "]" ||
    ch === "}" ||
    ch === "」" ||
    ch === "』" ||
    ch === "》" ||
    ch === "›" ||
    ch === "»";

  const isSkippableBeforeCitation = (ch: string) =>
    isBoundary(ch) || isTrailingWrapper(ch) || /\s/.test(ch);

  // 检查是否出现在段首（前面只有空白字符或换行符）
  let checkPos = startFrom;
  while (checkPos > 0 && /\s/.test(content[checkPos - 1])) {
    checkPos -= 1;
  }
  // 如果前面遇到换行符或到达文本开头，说明在段首
  if (checkPos === 0 || content[checkPos - 1] === "\n") {
    return "";
  }

  let searchEnd = startFrom;
  while (searchEnd > 0 && isSkippableBeforeCitation(content[searchEnd - 1])) {
    searchEnd -= 1;
  }

  // 向左查找，遇到换行符或引用标记则停止，最多跨越10个句子边界，或500字
  let left = searchEnd;
  let sentenceCount = 0;
  const maxSentences = 10;
  const maxChars = 500;
  let hasCitationBoundary = false;

  while (left > 0) {
    // 检查当前位置之前是否有引用标记
    const checkStart = Math.max(0, left - 10);
    const checkText = content.slice(checkStart, left);
    const citeMatch = checkText.match(/【\d+】$/);
    if (citeMatch) {
      // 遇到引用标记，设置left为标记结束位置
      left = checkStart + citeMatch.index! + citeMatch[0].length;
      hasCitationBoundary = true;
      break;
    }

    const prevChar = content[left - 1];

    // 遇到换行符，立即停止
    if (prevChar === "\n") {
      break;
    }

    // 遇到句子边界，计数
    if (isBoundary(prevChar)) {
      sentenceCount++;
      // 如果已经遇到引用标记边界，或者句子数量超过5，则停止
      if (hasCitationBoundary || sentenceCount >= maxSentences) {
        break;
      }
    }

    // 检查当前提取内容的长度
    const currentLength = startFrom - left;
    if (currentLength >= maxChars) {
      break;
    }

    left -= 1;
  }

  // 右边界仍然用 startFrom，保留句末标点（如果存在）
  const beforeMark = stripCitationMarkers(
    content.slice(left, startFrom)
  ).trim();
  return beforeMark;
}

/**
 * 将 query 中的引用标记移除（支持多个引用）
 */
export function stripCitationMarkers(text: string): string {
  return (text || "").replace(/【\d+】/g, "").trim();
}

function normalizeForNgram(text: string): string {
  return (
    (text || "")
      .replace(/<[^>]+>/g, "")
      .toLowerCase()
      .replace(/\s+/g, "")
      // 常见中英文标点与符号去噪（保留中英文与数字）
      .replace(
        /[，,。.!！？?；;：:""""''（）()\[\]【】<>《》、/\\|@#$%^&*_+=\-~`]/g,
        ""
      )
  );
}

function buildBigrams(text: string): Set<string> {
  const normalized = normalizeForNgram(text);
  const grams = new Set<string>();
  for (let i = 0; i <= normalized.length - 2; i += 1) {
    grams.add(normalized.slice(i, i + 2));
  }
  return grams;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  small.forEach((t) => {
    if (big.has(t)) inter += 1;
  });
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

function coverageOfQuery(query: Set<string>, cand: Set<string>): number {
  if (query.size === 0) return 0;
  let inter = 0;
  query.forEach((t) => {
    if (cand.has(t)) inter += 1;
  });
  return inter / query.size;
}

function splitToClauses(text: string): string[] {
  const raw = (text || "").replace(/\r\n/g, "\n").trim();
  if (!raw) return [];

  // 用"句末标点/换行"切分，同时尽量保留标点到子句末尾，保证可读性
  const clauses: string[] = [];
  let buf = "";
  const pushBuf = () => {
    const s = buf.trim();
    if (s) clauses.push(s);
    buf = "";
  };

  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    buf += ch;
    const isEnd = isBoundary(ch);
    if (isEnd) pushBuf();
  }
  pushBuf();

  // 去掉过短子句（避免噪声）
  return clauses.filter((c) => normalizeForNgram(c).length >= 6);
}

/**
 * 在 source（引文全文）里找与 query 最匹配的片段
 * @param source - 引文全文 mapData.content
 * @param query - 来自 markdown 的"包含引用标记的句子"去标记后的文本
 * @param options - 可选参数
 */
export function findBestCitationExcerpt(
  source: string,
  query: string,
  options?: {
    /** 允许横跨的最大子句数量，默认 3 */
    maxSpan?: 1 | 2 | 3 | 4 | 5;
    /** 返回片段的最大长度，超出会裁剪加省略号，默认 280 */
    maxExcerptLength?: number;
  }
): CitationBestMatch | null {
  const { maxSpan = 3, maxExcerptLength = 280 } = options || {};

  // early stop：达到足够高的匹配度就提前结束，避免长文档全量扫描
  const earlyStopThreshold = 0.75;

  const qNorm = normalizeForNgram(query);
  if (!qNorm || qNorm.length < 2) return null;

  const clauses = splitToClauses(source);
  if (clauses.length === 0) return null;

  // 候选：1/2/3/4/5 子句拼接（相邻）
  const candidates: { text: string; clauseIndex: number; span: number }[] = [];
  for (let i = 0; i < clauses.length; i += 1) {
    // 生成不同跨度的子句组合
    for (
      let span = 1;
      span <= Math.min(maxSpan, clauses.length - i);
      span += 1
    ) {
      const combinedText = clauses.slice(i, i + span).join("");
      candidates.push({
        text: combinedText,
        clauseIndex: i,
        span,
      });
    }
  }

  // 按子句长度排序，保证返回时子句最短
  candidates.sort((a, b) => a.span - b.span);

  // 规则优先：包含关系命中直接返回最高分（常见于引文包含原句）
  for (const c of candidates) {
    const cNorm = normalizeForNgram(c.text);
    if (!cNorm) continue;
    if (cNorm.includes(qNorm) || qNorm.includes(cNorm)) {
      const finalText = clampExcerpt(c.text, maxExcerptLength);
      return {
        text: finalText,
        score: 1,
        clauseIndex: c.clauseIndex,
        span: c.span,
      };
    }
  }

  // bigram Jaccard + coverage
  const qBi = buildBigrams(query);
  if (qBi.size === 0) return null;

  let best: CitationBestMatch = {
    text: candidates[0].text,
    score: 0,
    clauseIndex: candidates[0].clauseIndex,
    span: candidates[0].span,
  };

  for (const c of candidates) {
    const cBi = buildBigrams(c.text);
    const jac = jaccard(qBi, cBi);
    const cov = coverageOfQuery(qBi, cBi);
    /**
     * 说明：引文候选往往比 query 长很多，Jaccard 会因 union 巨大而偏小；
     * 我们更关心"query 被候选覆盖得有多好"，因此 coverage 权重更高
     */
    const score = 0.2 * jac + 0.8 * cov;
    if (score > best.score) {
      best = { text: c.text, score, clauseIndex: c.clauseIndex, span: c.span };
    }

    // 达到阈值则提前结束（越早命中越节省计算）
    if (best.score >= earlyStopThreshold) break;
  }

  /**
   * 返回策略：只要找到"非 0 分"的最佳候选就返回，
   * 避免在业务侧回退到整篇引文导致 tooltip 过长
   */
  if (best.score <= 0) return null;
  return { ...best, text: clampExcerpt(best.text, maxExcerptLength) };
}

function clampExcerpt(text: string, maxLen: number): string {
  const raw = (text || "").trim();
  if (!raw) return "";
  if (raw.length <= maxLen) return raw;
  return `${raw.slice(0, maxLen)}…`;
}
