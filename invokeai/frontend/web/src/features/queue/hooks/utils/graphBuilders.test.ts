import { describe, expect, it, vi } from 'vitest';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import type { RootState } from 'app/store/store';

const mocks = vi.hoisted(() => {
  const mockGraph: Graph = {} as Graph;
  const mockPrompt = { id: 'prompt-node' } as Invocation<'string'>;
  const asyncReturnValue = { g: mockGraph, positivePrompt: mockPrompt };
  const syncReturnValue = { g: mockGraph, positivePrompt: mockPrompt };

  return {
    asyncReturnValue,
    syncReturnValue,
    buildSDXLGraphMock: vi.fn().mockResolvedValue(asyncReturnValue),
    buildImagen3GraphMock: vi.fn().mockReturnValue(syncReturnValue),
    createDefaultBuilder: () => vi.fn().mockResolvedValue(asyncReturnValue),
  };
});

vi.mock('features/nodes/util/graph/generation/buildSDXLGraph', () => ({
  buildSDXLGraph: mocks.buildSDXLGraphMock,
}));
vi.mock('features/nodes/util/graph/generation/buildSD1Graph', () => ({
  buildSD1Graph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildSD3Graph', () => ({
  buildSD3Graph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildFLUXGraph', () => ({
  buildFLUXGraph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildFluxKontextGraph', () => ({
  buildFluxKontextGraph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildCogView4Graph', () => ({
  buildCogView4Graph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildImagen3Graph', () => ({
  buildImagen3Graph: mocks.buildImagen3GraphMock,
}));
vi.mock('features/nodes/util/graph/generation/buildImagen4Graph', () => ({
  buildImagen4Graph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildChatGPT4oGraph', () => ({
  buildChatGPT4oGraph: mocks.createDefaultBuilder(),
}));
vi.mock('features/nodes/util/graph/generation/buildGemini2_5Graph', () => ({
  buildGemini2_5Graph: mocks.createDefaultBuilder(),
}));

import { buildGraphForBase } from './graphBuilders';

describe('buildGraphForBase', () => {
  const baseArg: GraphBuilderArg = {
    generationMode: 'txt2img',
    state: {} as RootState,
    manager: null,
  };

  it('awaits asynchronous graph builders', async () => {
    const result = await buildGraphForBase('sdxl', baseArg);

    expect(result).toBe(mocks.asyncReturnValue);
    expect(mocks.buildSDXLGraphMock).toHaveBeenCalledWith(baseArg);
  });

  it('supports synchronous graph builders', async () => {
    const result = await buildGraphForBase('imagen3', baseArg);

    expect(result).toBe(mocks.syncReturnValue);
    expect(mocks.buildImagen3GraphMock).toHaveBeenCalledWith(baseArg);
  });

  it('throws for unknown base models', async () => {
    await expect(buildGraphForBase('unknown-model', baseArg)).rejects.toThrow(
      'No graph builders for base unknown-model'
    );
  });
});
