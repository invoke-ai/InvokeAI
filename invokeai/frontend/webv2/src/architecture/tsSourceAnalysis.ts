import type { Node, SourceFile, Statement } from 'typescript/unstable/ast';

import {
  isCallExpression,
  isClassDeclaration,
  isEmptyStatement,
  isEnumDeclaration,
  isExportAssignment,
  isExportDeclaration,
  isFunctionDeclaration,
  isIdentifier,
  isImportDeclaration,
  isImportTypeNode,
  isInterfaceDeclaration,
  isLiteralTypeNode,
  isNamedExports,
  isNamedImports,
  isNamespaceExport,
  isNamespaceImport,
  isStringLiteralLikeNode,
  isTypeAliasDeclaration,
  isVariableStatement,
  ModifierFlags,
  SyntaxKind,
} from 'typescript/unstable/ast';
import { createVirtualFileSystem } from 'typescript/unstable/fs';
import { API } from 'typescript/unstable/sync';

export interface SourceAnalysisOptions {
  jsx?: boolean;
}

export type SourceModuleReferenceForm = 'dynamic-import' | 'export-declaration' | 'import-declaration' | 'import-type';

export type SourceModuleReferenceKind = 'dynamic-import' | 'export' | 'export-star' | 'import' | 'import-type';

export interface SourceModuleReference {
  form: SourceModuleReferenceForm;
  kind: SourceModuleReferenceKind;
  namespace: boolean;
  qualifier: string | null;
  specifier: string;
  symbols: readonly string[];
}

export interface SourceAnalysis {
  moduleReferences: readonly SourceModuleReference[];
  publicExports: readonly string[];
  typeOnly: boolean;
}

const PARSE_ROOT = '/architecture-sources';
const CONFIG_PATH = `${PARSE_ROOT}/tsconfig.json`;
const SEED_PATH = `${PARSE_ROOT}/seed.tsx`;
const CONFIG = JSON.stringify({
  compilerOptions: { allowJs: true, jsx: 'preserve', module: 'esnext', noLib: true, target: 'esnext' },
  include: ['**/*'],
});

let fileSystem: ReturnType<typeof createVirtualFileSystem> | undefined;
let api: API | undefined;
let snapshot: ReturnType<API['updateSnapshot']> | undefined;
let nextId = 0;
const virtualPaths = new Map<string, string>();
const stagedText = new Map<string, string>();
const created: string[] = [];
const changed: string[] = [];

const ensureSession = (): void => {
  if (!api) {
    fileSystem = createVirtualFileSystem({ [CONFIG_PATH]: CONFIG, [SEED_PATH]: 'export {};' });
    api = new API({ cwd: PARSE_ROOT, fs: fileSystem });
    snapshot = api.updateSnapshot({ openProjects: [CONFIG_PATH] });
  }
};

const resolveJsx = (fileName: string, jsx: boolean | undefined): boolean => jsx ?? fileName.endsWith('x');

const virtualPathFor = (fileName: string, jsx: boolean): string => {
  const key = `${jsx ? 'x' : ''}:${fileName}`;
  let virtualPath = virtualPaths.get(key);

  if (!virtualPath) {
    virtualPath = `${PARSE_ROOT}/${nextId++}.${jsx ? 'tsx' : 'ts'}`;
    virtualPaths.set(key, virtualPath);
  }

  return virtualPath;
};

const stage = (fileName: string, text: string, jsx: boolean): string => {
  ensureSession();
  const virtualPath = virtualPathFor(fileName, jsx);

  if (stagedText.get(virtualPath) === text) {
    return virtualPath;
  }

  (stagedText.has(virtualPath) ? changed : created).push(virtualPath);
  stagedText.set(virtualPath, text);
  fileSystem?.writeFile?.(virtualPath, text);
  return virtualPath;
};

const flush = (): void => {
  if (created.length === 0 && changed.length === 0) {
    return;
  }
  const previousSnapshot = snapshot;
  snapshot = api?.updateSnapshot({
    fileChanges: { changed: [...changed], created: [...created] },
    openProjects: [CONFIG_PATH],
  });
  previousSnapshot?.dispose();
  created.length = 0;
  changed.length = 0;
};

const read = (virtualPath: string, fileName: string): SourceFile => {
  const sourceFile = snapshot?.getProjects()[0]?.program.getSourceFile(virtualPath);

  if (!sourceFile) {
    throw new Error(`Failed to parse ${fileName} through the TypeScript API.`);
  }

  return sourceFile;
};

const hasExportModifier = (statement: Statement): boolean => {
  const { modifierFlags } = statement as { modifierFlags?: ModifierFlags };
  return modifierFlags !== undefined && (modifierFlags & ModifierFlags.Export) !== 0;
};

const collectPublicExports = (sourceFile: SourceFile): string[] => {
  const exports = new Set<string>();

  for (const statement of sourceFile.statements) {
    if (hasExportModifier(statement)) {
      if (
        (isClassDeclaration(statement) ||
          isFunctionDeclaration(statement) ||
          isInterfaceDeclaration(statement) ||
          isTypeAliasDeclaration(statement) ||
          isEnumDeclaration(statement)) &&
        statement.name
      ) {
        exports.add(statement.name.text);
      } else if (isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (isIdentifier(declaration.name)) {
            exports.add(declaration.name.text);
          }
        }
      }
    }

    if (isExportAssignment(statement)) {
      exports.add('default');
    } else if (isExportDeclaration(statement)) {
      if (!statement.exportClause) {
        exports.add('*');
      } else if (isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) {
          exports.add(element.name.text);
        }
      }
    }
  }

  return [...exports].sort();
};

const isTypeOnlyModule = (sourceFile: SourceFile): boolean => {
  return sourceFile.statements.every((statement) => {
    if (isInterfaceDeclaration(statement) || isTypeAliasDeclaration(statement)) {
      return true;
    }
    if (isImportDeclaration(statement)) {
      return statement.importClause?.phaseModifier === SyntaxKind.TypeKeyword;
    }
    if (isExportDeclaration(statement)) {
      return statement.isTypeOnly;
    }
    return isEmptyStatement(statement);
  });
};

const collectModuleReferences = (sourceFile: SourceFile): SourceModuleReference[] => {
  const references: SourceModuleReference[] = [];

  const visit = (node: Node): void => {
    if (isImportDeclaration(node) && node.moduleSpecifier && isStringLiteralLikeNode(node.moduleSpecifier)) {
      const symbols: string[] = [];
      const bindings = node.importClause?.namedBindings;
      const namespace = Boolean(bindings && isNamespaceImport(bindings));

      if (node.importClause?.name) {
        symbols.push('default');
      }
      if (bindings && isNamespaceImport(bindings)) {
        symbols.push('*');
      } else if (bindings && isNamedImports(bindings)) {
        symbols.push(...bindings.elements.map((element) => (element.propertyName ?? element.name).text));
      }

      references.push({
        form: 'import-declaration',
        kind: node.importClause?.phaseModifier === SyntaxKind.TypeKeyword ? 'import-type' : 'import',
        namespace,
        qualifier: null,
        specifier: node.moduleSpecifier.text,
        symbols,
      });
    } else if (isExportDeclaration(node) && node.moduleSpecifier && isStringLiteralLikeNode(node.moduleSpecifier)) {
      const namespace = !node.exportClause || isNamespaceExport(node.exportClause);
      const symbols = namespace
        ? ['*']
        : isNamedExports(node.exportClause)
          ? node.exportClause.elements.map((element) => (element.propertyName ?? element.name).text)
          : [];

      references.push({
        form: 'export-declaration',
        kind: namespace ? 'export-star' : 'export',
        namespace,
        qualifier: null,
        specifier: node.moduleSpecifier.text,
        symbols,
      });
    } else if (
      isImportTypeNode(node) &&
      isLiteralTypeNode(node.argument) &&
      isStringLiteralLikeNode(node.argument.literal)
    ) {
      references.push({
        form: 'import-type',
        kind: 'import-type',
        namespace: false,
        qualifier: node.qualifier?.getText(sourceFile) ?? null,
        specifier: node.argument.literal.text,
        symbols: [],
      });
    } else if (
      isCallExpression(node) &&
      node.expression.kind === SyntaxKind.ImportKeyword &&
      node.arguments[0] &&
      isStringLiteralLikeNode(node.arguments[0])
    ) {
      references.push({
        form: 'dynamic-import',
        kind: 'dynamic-import',
        namespace: false,
        qualifier: null,
        specifier: node.arguments[0].text,
        symbols: [],
      });
    }

    node.forEachChild(visit);
  };

  visit(sourceFile);
  return references;
};

/** Stages many sources so a later analysis sweep shares one snapshot update. */
export const primeSourceAnalysis = (
  entries: Iterable<readonly [string, string]>,
  { jsx }: SourceAnalysisOptions = {}
): void => {
  for (const [fileName, text] of entries) {
    stage(fileName, text, resolveJsx(fileName, jsx));
  }
  flush();
};

export const analyzeSource = (fileName: string, text: string, { jsx }: SourceAnalysisOptions = {}): SourceAnalysis => {
  const virtualPath = stage(fileName, text, resolveJsx(fileName, jsx));
  flush();
  const sourceFile = read(virtualPath, fileName);

  return {
    moduleReferences: collectModuleReferences(sourceFile),
    publicExports: collectPublicExports(sourceFile),
    typeOnly: isTypeOnlyModule(sourceFile),
  };
};

export const closeSourceAnalysis = (): void => {
  snapshot?.dispose();
  api?.close();
  fileSystem = undefined;
  api = undefined;
  snapshot = undefined;
  nextId = 0;
  virtualPaths.clear();
  stagedText.clear();
  created.length = 0;
  changed.length = 0;
};
