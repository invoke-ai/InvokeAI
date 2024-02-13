/**
 * Invalid Workflow Version Error
 * Raised when a workflow version is not recognized.
 */
export class WorkflowVersionError extends Error {
  /**
   * Create WorkflowVersionError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}
/**
 * Workflow Migration Error
 * Raised when a workflow migration fails.
 */
export class WorkflowMigrationError extends Error {
  /**
   * Create WorkflowMigrationError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

/**
 * Unable to Update Node Error
 * Raised when a node cannot be updated.
 */
export class NodeUpdateError extends Error {
  /**
   * Create NodeUpdateError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

/**
 * FieldParseError
 * Raised when a field cannot be parsed from a field schema.
 */
export class FieldParseError extends Error {
  /**
   * Create FieldTypeParseError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}
