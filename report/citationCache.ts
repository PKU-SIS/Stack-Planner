/**
 * Report 引用匹配结果缓存
 *
 * 用于流式输出场景，避免同一引用句（citeId + querySentence）重复执行
 * 数字匹配、文字匹配、全量重排等耗时逻辑。采用 LRU 策略控制内存占用。
 */

export type CitationCacheValue = string;

/**
 * 简单 LRU 缓存，避免同一引用句重复匹配
 * @param limit 最大缓存条目数，超出时淘汰最久未访问的项
 */
export class CitationResultCache {
  private readonly limit: number;
  private readonly map = new Map<string, CitationCacheValue>();

  constructor(limit: number = 1000) {
    this.limit = Math.max(1, limit);
  }

  /** 获取缓存值，命中时将该键移至最近使用 */
  get(key: string): CitationCacheValue | undefined {
    const value = this.map.get(key);
    if (value === undefined) return undefined;
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  /** 写入缓存，若已存在则更新；超出容量时淘汰最久未访问项 */
  set(key: string, value: CitationCacheValue): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size <= this.limit) return;
    const firstKey = this.map.keys().next().value;
    if (firstKey !== undefined) this.map.delete(firstKey);
  }

  /** 清空缓存 */
  clear(): void {
    this.map.clear();
  }
}

/**
 * 构建引用缓存键，用于去重同一引用在不同上下文的重复计算
 * @param citeId 引用标记中的数字 ID，如 "1"
 * @param querySentence 正文中提取的查询句（已去除引用标记）
 */
export function buildCitationCacheKey(citeId: string, querySentence: string): string {
  return `${citeId}|${hashText(querySentence)}`;
}

/** FNV-1a 风格简单哈希，用于 querySentence 的短指纹 */
function hashText(input: string): string {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}
