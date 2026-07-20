import { brotliCompressSync, constants, gzipSync } from 'node:zlib';

const collectStaticImports = (manifest, source, collected = new Set()) => {
  if (collected.has(source)) {
    return collected;
  }
  const chunk = manifest[source];
  if (!chunk) {
    throw new Error(`Build manifest is missing route source "${source}".`);
  }
  collected.add(source);
  for (const imported of chunk.imports ?? []) {
    collectStaticImports(manifest, imported, collected);
  }
  return collected;
};

const getChunkName = (source, chunk) => chunk.name ?? chunk.src ?? source;

export const measureRouteBuild = (manifest, routeId, source, readAsset) => {
  const sources = [...collectStaticImports(manifest, source)];
  const assets = sources.map((chunkSource) => {
    const chunk = manifest[chunkSource];
    return { bytes: readAsset(chunk.file), chunk, source: chunkSource };
  });
  const owned = assets.find((asset) => asset.source === source);

  return {
    brotliBytes: assets.reduce(
      (total, asset) =>
        total + brotliCompressSync(asset.bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: 4 } }).byteLength,
      0
    ),
    chunkNames: assets.map((asset) => getChunkName(asset.source, asset.chunk)).sort(),
    files: assets.map((asset) => asset.chunk.file).sort(),
    gzipBytes: assets.reduce((total, asset) => total + gzipSync(asset.bytes, { level: 9 }).byteLength, 0),
    initialRawBytes: assets.reduce((total, asset) => total + asset.bytes.byteLength, 0),
    ownedRawBytes: owned.bytes.byteLength,
    routeId,
    source,
    sources: sources.sort(),
  };
};

export const checkRouteBudget = (measurement, budget) => {
  const failures = [];
  const allowedGrowth = Math.min(
    budget.maxGrowthRawBytes,
    Math.floor(budget.baselineOwnedRawBytes * budget.maxGrowthPercent)
  );
  const maximumOwnedBytes = budget.baselineOwnedRawBytes + allowedGrowth;

  if (measurement.ownedRawBytes > maximumOwnedBytes) {
    failures.push({
      actual: measurement.ownedRawBytes,
      budget: maximumOwnedBytes,
      message: `${measurement.routeId} owned JavaScript grew to ${measurement.ownedRawBytes} bytes (budget ${maximumOwnedBytes}, baseline ${budget.baselineOwnedRawBytes}).`,
      owner: budget.owner,
      remediationTicket: budget.remediationTicket,
      routeId: measurement.routeId,
    });
  }
  if (JSON.stringify(measurement.chunkNames) !== JSON.stringify(budget.initialChunkNames)) {
    failures.push({
      actual: measurement.chunkNames,
      budget: budget.initialChunkNames,
      message: `${measurement.routeId} initial request set changed. Expected [${budget.initialChunkNames.join(', ')}], received [${measurement.chunkNames.join(', ')}].`,
      owner: budget.owner,
      remediationTicket: budget.remediationTicket,
      routeId: measurement.routeId,
    });
  }
  return failures;
};
