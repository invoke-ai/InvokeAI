/**
 * Raised when metadata parsing fails.
 */
export class MetadataParseError extends Error {
  /**
   * Create MetadataParseError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

/**
 * Raised when metadata recall fails.
 */
export class MetadataRecallError extends Error {
  /**
   * Create MetadataRecallError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}
