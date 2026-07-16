const FULL_ENGINE_MODULE_PATTERN = String.raw`(?:@workbench/canvas-engine/engine|@workbench/canvas-operations/createCanvasEngine)`;
const FULL_ENGINE_MODULE = new RegExp(String.raw`['"]${FULL_ENGINE_MODULE_PATTERN}['"]`);

/** Detects every supported syntax that exposes or references the full CanvasEngine type. */
export const referencesFullCanvasEngine = (source: string): boolean => {
  const namedReference = new RegExp(
    String.raw`(?:import|export)(?:\s+type)?\s*\{[^}]*\bCanvasEngine\b[^}]*\}\s*from\s*['"]${FULL_ENGINE_MODULE_PATTERN}['"]`,
    's'
  );
  if (namedReference.test(source)) {
    return true;
  }

  const starReExport = new RegExp(
    String.raw`export\s*\*\s*(?:as\s+[A-Za-z_$][\w$]*\s*)?from\s*['"]${FULL_ENGINE_MODULE_PATTERN}['"]`
  );
  if (starReExport.test(source)) {
    return true;
  }

  const inlineReference = new RegExp(
    String.raw`import\s*\(\s*['"]${FULL_ENGINE_MODULE_PATTERN}['"]\s*\)\s*\.\s*CanvasEngine\b`
  );
  if (inlineReference.test(source)) {
    return true;
  }

  const namespaceImport = /import(?:\s+type)?\s*\*\s*as\s*[A-Za-z_$][\w$]*\s*from\s*([^;\n]+)/g;
  for (const match of source.matchAll(namespaceImport)) {
    const moduleSource = match[1];
    if (moduleSource && FULL_ENGINE_MODULE.test(moduleSource)) {
      return true;
    }
  }

  return false;
};
