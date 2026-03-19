/**
 * Report 数字匹配工具
 *
 * 功能：
 * - 将中文数字转换为阿拉伯数字
 * - 基于数字匹配引文和正文内容
 * - 基于引文和正文中的数字判断是否匹配
 *
 * 与 citationMatch（文字相似度）配合使用，优先数字匹配，未命中时回退到文字匹配。
 */

export type NumberBestMatch = {
  /** 最佳匹配片段（原文） */
  text: string;
  /** 置信分数（0~1） */
  score: number;
  /** 起始子句 index */
  clauseIndex: number;
  /** 横跨子句数量（1/2/3） */
  span: number;
  /** 匹配的数字 */
  matchNumber: number[];
};

// 中文数字映射
const CHINESE_NUMBERS: Record<string, number> = {
  零: 0,
  一: 1,
  二: 2,
  三: 3,
  四: 4,
  五: 5,
  六: 6,
  七: 7,
  八: 8,
  九: 9,
  十: 10,
  百: 100,
  千: 1000,
  万: 10000,
  亿: 100000000,
};

// 中文约数词（如 500余万、500多万），解析时过滤，按基数处理
const APPROXIMATE_WORDS = /[余多]/g;

// 阿拉伯数字后缀单位映射（用于"500万"、"1.5亿"这类写法）
const ARABIC_UNIT_MULTIPLIER: Record<string, number> = {
  十: 10,
  百: 100,
  千: 1000,
  万: 10000,
  亿: 100000000,
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

/** 按句末标点/换行切分子句，过滤过短子句 */
function splitToClauses(text: string): string[] {
  const raw = (text || "").replace(/\r\n/g, "\n").trim();
  if (!raw) return [];

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
    if (isBoundary(ch)) pushBuf();
  }
  pushBuf();

  return clauses.filter((c) => c.length >= 6);
}

/**
 * 将中文数字转换为阿拉伯数字
 * @param chineseNum 中文数字字符串，如 "五万四千三百点二五"
 * @returns 阿拉伯数字，如 54300.25
 */
export function chineseToArabic(chineseNum: string): number | null {
  if (!chineseNum || typeof chineseNum !== "string") return null;

  try {
    const stripped = chineseNum.replace(APPROXIMATE_WORDS, "");
    const parts = stripped.split("点");
    if (parts.length > 2) return null;

    let integerPart = parts[0];
    let decimalPart = parts[1] || "";

    const integerValue = parseChineseInteger(integerPart);
    if (integerValue === null) return null;

    let decimalValue = 0;
    if (decimalPart) {
      const decimalResult = parseChineseDecimal(decimalPart);
      if (decimalResult === null) return null;
      decimalValue = decimalResult;
    }

    return integerValue + decimalValue;
  } catch {
    return null;
  }
}

/** 解析中文整数部分 */
function parseChineseInteger(chinese: string): number | null {
  if (!chinese) return 0;

  let result = 0;
  let temp = 0;
  let lastUnit = 1;

  for (let i = 0; i < chinese.length; i++) {
    const char = chinese[i];
    const num = CHINESE_NUMBERS[char];

    if (num === undefined) return null;

    if (char === "十") {
      if (i === 0 || CHINESE_NUMBERS[chinese[i - 1]] === undefined) {
        temp = 10;
      } else {
        temp *= 10;
        lastUnit = 10;
      }
    } else if (
      char === "百" ||
      char === "千" ||
      char === "万" ||
      char === "亿"
    ) {
      if (temp === 0 && num > 10) {
        temp = num / 10;
      } else {
        temp *= num;
      }
      result += temp;
      temp = 0;
      lastUnit = num;
    } else if (num >= 0 && num <= 9) {
      if (temp === 0) {
        temp = num;
      } else {
        temp += num;
      }
    } else {
      return null;
    }
  }

  result += temp;
  return result;
}

/** 解析中文小数部分 */
function parseChineseDecimal(chinese: string): number | null {
  if (!chinese) return 0;

  let result = 0;
  let denominator = 10;

  for (const char of chinese) {
    const num = CHINESE_NUMBERS[char];
    if (num === undefined || num < 0 || num > 9) return null;

    result += num / denominator;
    denominator *= 10;
  }

  return result;
}

/**
 * 解析阿拉伯数字写法，支持千分位、中文单位后缀（500万 / 1.5亿）
 */
function parseArabicLikeNumber(raw: string): number | null {
  if (!raw) return null;

  const normalized = raw
    .replace(/,/g, "")
    .replace(APPROXIMATE_WORDS, "");
  const unitMatch = normalized.match(/^(\d+(?:\.\d+)?)([十百千万亿])?$/);
  if (!unitMatch) return null;

  const base = parseFloat(unitMatch[1]);
  if (Number.isNaN(base)) return null;

  const unit = unitMatch[2];
  const multiplier = unit ? ARABIC_UNIT_MULTIPLIER[unit] : 1;
  if (!multiplier) return null;

  return base * multiplier;
}

/**
 * 从文本中提取所有数字（阿拉伯数字和中文数字），归一化后便于对齐匹配
 */
export function extractNumbers(
  text: string
): Array<{ value: number; original: string; start: number; end: number }> {
  const numbers: Array<{
    value: number;
    original: string;
    start: number;
    end: number;
  }> = [];

  if (!text) return numbers;

  const occupiedRanges = new Set<number>();

  const arabicRegex =
    /(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[余多])?(?:[十百千万亿])?/g;
  let match: RegExpExecArray | null;
  while ((match = arabicRegex.exec(text)) !== null) {
    const value = parseArabicLikeNumber(match[0]);
    if (value !== null) {
      numbers.push({
        value,
        original: match[0],
        start: match.index,
        end: match.index + match[0].length,
      });
      for (let i = match.index; i < match.index + match[0].length; i++) {
        occupiedRanges.add(i);
      }
    }
  }

  const chineseRegex = /[零一二三四五六七八九十百千万亿点余多]+/g;
  while ((match = chineseRegex.exec(text)) !== null) {
    const matchText = match[0];
    const matchIndex = match.index;

    const isOccupied = Array.from({ length: matchText.length }, (_, i) =>
      occupiedRanges.has(matchIndex + i)
    ).some(Boolean);

    if (!isOccupied) {
      const arabic = chineseToArabic(matchText);
      if (arabic !== null) {
        numbers.push({
          value: arabic,
          original: matchText,
          start: matchIndex,
          end: matchIndex + matchText.length,
        });
      }
    }
  }

  return numbers;
}

/** 过滤日期相关数字（紧邻年月日） */
function isDateRelatedNumber(
  text: string,
  numberInfo: { value: number; original: string; start: number; end: number }
): boolean {
  const value = numberInfo.value;
  const after = text[numberInfo.end] || "";

  if (after === "年" && value >= 1900 && value <= 2099) return true;
  if (after === "月" && value >= 1 && value <= 12) return true;
  if ((after === "日" || after === "号") && value >= 1 && value <= 31) return true;

  const contextStart = Math.max(0, numberInfo.start - 10);
  const contextEnd = Math.min(text.length, numberInfo.end + 10);
  const context = text.slice(contextStart, contextEnd);
  const fullDatePattern = /\d{2,4}年\d{1,2}月\d{1,2}[日号]/;

  if (fullDatePattern.test(context)) {
    const dateMatch = context.match(fullDatePattern);
    if (dateMatch) {
      const dateStr = dateMatch[0];
      const dateStart = context.indexOf(dateStr);
      const dateEnd = dateStart + dateStr.length;
      const relativeStart = numberInfo.start - contextStart;
      const relativeEnd = numberInfo.end - contextStart;

      if (relativeStart >= dateStart && relativeEnd <= dateEnd) {
        return true;
      }
    }
  }

  return false;
}

/**
 * 基于数字匹配找到最佳片段
 * 优先匹配正文与引文中的数字，排除日期类数字
 */
export function findBestNumberExcerpt(
  source: string,
  query: string,
  options: {
    maxSpan?: number;
    maxExcerptLength?: number;
  }
): NumberBestMatch | null {
  const { maxSpan = 3, maxExcerptLength = 280 } = options || {};

  if (!source || !query) return null;

  const queryNumbers = extractNumbers(query);
  if (queryNumbers.length === 0) return null;

  const validQueryNumbers = queryNumbers.filter(
    (n) => n.original.length >= 2 && !isDateRelatedNumber(query, n)
  );

  if (validQueryNumbers.length === 0) return null;

  const clauses = splitToClauses(source);
  if (clauses.length === 0) return null;

  const clauseNumbers = clauses.map((clause) =>
    extractNumbers(clause).filter(
      (n) => n.original.length >= 2 && !isDateRelatedNumber(clause, n)
    )
  );

  const candidates: {
    text: string;
    clauseIndex: number;
    span: number;
    numbers: Array<{
      value: number;
      original: string;
      start: number;
      end: number;
    }>;
  }[] = [];

  for (let i = 0; i < clauses.length; i += 1) {
    for (
      let span = 1;
      span <= Math.min(maxSpan, clauses.length - i);
      span += 1
    ) {
      const combinedText = clauses.slice(i, i + span).join("");
      const combinedNumbers: Array<{
        value: number;
        original: string;
        start: number;
        end: number;
      }> = [];
      const seenValues = new Set<number>();

      for (let j = i; j < i + span; j++) {
        for (const num of clauseNumbers[j]) {
          if (!seenValues.has(num.value)) {
            combinedNumbers.push(num);
            seenValues.add(num.value);
          }
        }
      }

      candidates.push({
        text: combinedText,
        clauseIndex: i,
        span,
        numbers: combinedNumbers,
      });
    }
  }

  candidates.sort((a, b) => a.span - b.span);

  let bestMatch: NumberBestMatch | null = null;
  let bestMatchCount = 0;

  for (const candidate of candidates) {
    let matchCount = 0;
    const matchedNumbers: number[] = [];

    for (const queryNum of validQueryNumbers) {
      for (const candNum of candidate.numbers) {
        if (queryNum.value === candNum.value) {
          matchCount++;
          matchedNumbers.push(candNum.value);
          break;
        }
      }
    }

    if (matchCount === validQueryNumbers.length) {
      return {
        text:
          candidate.text.length > maxExcerptLength
            ? candidate.text.slice(0, maxExcerptLength) + "…"
            : candidate.text,
        score: 1.0,
        clauseIndex: candidate.clauseIndex,
        span: candidate.span,
        matchNumber: matchedNumbers,
      };
    }

    if (matchCount > bestMatchCount) {
      bestMatchCount = matchCount;
      bestMatch = {
        text:
          candidate.text.length > maxExcerptLength
            ? candidate.text.slice(0, maxExcerptLength) + "…"
            : candidate.text,
        score: matchCount / validQueryNumbers.length,
        clauseIndex: candidate.clauseIndex,
        span: candidate.span,
        matchNumber: matchedNumbers,
      };
    }
  }

  return bestMatch;
}
