import { zModelIdentifierField } from 'features/nodes/types/common';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// Mock model configs returned by selectors - these simulate what RTK Query provides
const mockAnimaQwen3Encoder = {
  key: 'qwen3-06b-key',
  hash: 'qwen3-06b-hash',
  name: 'Qwen3 0.6B Encoder',
  base: 'any' as const,
  type: 'qwen3_encoder' as const,
  variant: 'qwen3_06b' as const,
  format: 'qwen3_encoder' as const,
};

const mockAnimaVAE = {
  key: 'anima-vae-key',
  hash: 'anima-vae-hash',
  name: 'Anima VAE',
  base: 'anima' as const,
  type: 'vae' as const,
  format: 'diffusers' as const,
};

const mockAnimaMainModel = {
  key: 'anima-main-key',
  hash: 'anima-main-hash',
  name: 'Anima Generate',
  base: 'anima' as const,
  type: 'main' as const,
};

const mockFluxMainModel = {
  key: 'flux-main-key',
  hash: 'flux-main-hash',
  name: 'FLUX.1 Dev',
  base: 'flux' as const,
  type: 'main' as const,
};

const mockZImageTurboDiffusers = {
  key: 'zimage-turbo-diff-key',
  hash: 'zimage-turbo-diff-hash',
  name: 'Z-Image Turbo Diffusers',
  base: 'z-image' as const,
  type: 'main' as const,
  format: 'diffusers' as const,
  variant: 'turbo' as const,
};

const mockZImageZbaseDiffusers = {
  key: 'zimage-zbase-diff-key',
  hash: 'zimage-zbase-diff-hash',
  name: 'Z-Image Base Diffusers',
  base: 'z-image' as const,
  type: 'main' as const,
  format: 'diffusers' as const,
  variant: 'zbase' as const,
};

const mockZImageTurboMain = {
  key: 'zimage-turbo-main-key',
  hash: 'zimage-turbo-main-hash',
  name: 'Z-Image Turbo',
  base: 'z-image' as const,
  type: 'main' as const,
};

const mockZImageZbaseMain = {
  key: 'zimage-zbase-main-key',
  hash: 'zimage-zbase-main-hash',
  name: 'Z-Image Base',
  base: 'z-image' as const,
  type: 'main' as const,
};

const mockZImageTurboMainConfig = { ...mockZImageTurboMain, variant: 'turbo' as const };
const mockZImageZbaseMainConfig = { ...mockZImageZbaseMain, variant: 'zbase' as const };

// Track dispatched actions
const dispatched: Array<{ type: string; payload: unknown }> = [];
const mockDispatch = vi.fn((action: { type: string; payload: unknown }) => {
  dispatched.push(action);
});

// Mock logger
vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
  }),
}));

// Mock toast
vi.mock('features/toast/toast', () => ({
  toast: vi.fn(),
}));

// Mock i18next
vi.mock('i18next', () => ({
  t: (key: string) => key,
}));

// Mock model selectors from RTK Query hooks

const mockSelectAnimaQwen3EncoderModels = vi.fn((_state: unknown) => [mockAnimaQwen3Encoder]);

const mockSelectAnimaVAEModels = vi.fn((_state: unknown) => [mockAnimaVAE]);

const mockSelectZImageDiffusersModels = vi.fn((_state: unknown) => [] as unknown[]);

const mockSelectModelConfigsQuery = vi.fn((_state: unknown) => ({ data: undefined as unknown }));

const mockModelConfigsSelectById = vi.fn((_data: unknown, _key: string) => undefined as unknown);

vi.mock('services/api/hooks/modelsByType', () => ({
  selectAnimaQwen3EncoderModels: (state: unknown) => mockSelectAnimaQwen3EncoderModels(state),
  selectAnimaVAEModels: (state: unknown) => mockSelectAnimaVAEModels(state),
  selectQwen3EncoderModels: vi.fn(() => []),
  selectZImageDiffusersModels: (state: unknown) => mockSelectZImageDiffusersModels(state),
  selectFluxVAEModels: vi.fn(() => []),
  selectGlobalRefImageModels: vi.fn(() => []),
  selectRegionalRefImageModels: vi.fn(() => []),
}));

// Mock model configs adapter
vi.mock('services/api/endpoints/models', () => ({
  modelConfigsAdapterSelectors: {
    selectById: (data: unknown, key: string) => mockModelConfigsSelectById(data, key),
  },
  selectModelConfigsQuery: (state: unknown) => mockSelectModelConfigsQuery(state),
}));

vi.mock('services/api/types', () => ({
  isFluxKontextModelConfig: vi.fn(() => false),
  isFluxReduxModelConfig: vi.fn(() => false),
}));

// Mock canvas selectors
vi.mock('features/controlLayers/store/canvasStagingAreaSlice', () => ({
  buildSelectIsStaging: vi.fn(() => vi.fn(() => false)),
  selectCanvasSessionId: vi.fn(() => null),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectAllEntitiesOfType: vi.fn(() => []),
  selectBboxModelBase: vi.fn(() => 'anima'),
  selectCanvasSlice: vi.fn(() => ({})),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  refImageConfigChanged: vi.fn(),
  refImageModelChanged: vi.fn(),
  selectReferenceImageEntities: vi.fn(() => []),
}));

vi.mock('features/controlLayers/store/types', async () => {
  const actual = await vi.importActual('features/controlLayers/store/types');
  return {
    ...(actual as Record<string, unknown>),
    getEntityIdentifier: vi.fn(),
    isFlux2ReferenceImageConfig: vi.fn(() => false),
  };
});

vi.mock('features/controlLayers/store/util', () => ({
  initialFlux2ReferenceImage: {},
  initialFluxKontextReferenceImage: {},
  initialFLUXRedux: {},
  initialIPAdapter: {},
}));

vi.mock('features/modelManagerV2/models', () => ({
  SUPPORTS_REF_IMAGES_BASE_MODELS: ['sd-1', 'sdxl', 'flux', 'flux2'],
}));

vi.mock('features/controlLayers/store/canvasSlice', () => ({
  bboxSyncedToOptimalDimension: vi.fn(() => ({ type: 'bboxSyncedToOptimalDimension' })),
  rgRefImageModelChanged: vi.fn(),
}));

vi.mock('features/controlLayers/store/lorasSlice', () => ({
  loraIsEnabledChanged: vi.fn((payload: unknown) => ({ type: 'loraIsEnabledChanged', payload })),
}));

// Capture the listener effect so we can call it directly
let capturedEffect: ((action: unknown, api: unknown) => void) | null = null;

// Import actual action creators for assertion matching
const paramsSliceActual = (await vi.importActual('features/controlLayers/store/paramsSlice')) as {
  animaQwen3EncoderModelSelected: { type: string };
  animaVaeModelSelected: { type: string };
  zImageQwen3SourceModelSelected: { type: string };
};
const { animaQwen3EncoderModelSelected, animaVaeModelSelected, zImageQwen3SourceModelSelected } = paramsSliceActual;

// Import after mocks are set up
const { addModelSelectedListener } = await import('./modelSelected');
const { modelSelected } = await import('features/parameters/store/actions');
const { zParameterModel } = await import('features/parameters/types/parameterSchemas');

// Capture the effect
addModelSelectedListener(((config: { effect: typeof capturedEffect }) => {
  capturedEffect = config.effect;
}) as never);

function buildMockState(overrides: Record<string, unknown> = {}) {
  return {
    params: {
      model: null,
      vae: null,
      zImageVaeModel: null,
      zImageQwen3EncoderModel: null,
      zImageQwen3SourceModel: null,
      animaVaeModel: null,
      animaQwen3EncoderModel: null,
      animaScheduler: 'euler',
      flux2VaeModel: null,
      kleinQwen3EncoderModel: null,
      flux2DevMistralEncoderModel: null,
      zImageScheduler: 'euler',
      ...overrides,
    },
    loras: { loras: [] },
    canvas: {},
  };
}

describe('modelSelected listener - Anima defaulting', () => {
  beforeEach(() => {
    dispatched.length = 0;
    mockDispatch.mockClear();
    mockSelectAnimaQwen3EncoderModels.mockReturnValue([mockAnimaQwen3Encoder]);
    mockSelectAnimaVAEModels.mockReturnValue([mockAnimaVAE]);
  });

  it('should dispatch encoder models with full ModelIdentifierField payloads when switching to Anima', () => {
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockAnimaMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    // Find the dispatched actions for Anima encoders
    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    // Both should have been dispatched
    expect(qwen3Dispatch).toBeDefined();
    expect(vaeDispatch).toBeDefined();

    // The payloads must pass zModelIdentifierField validation (the actual schema used by reducers)
    expect(zModelIdentifierField.safeParse(qwen3Dispatch!.payload).success).toBe(true);
    expect(zModelIdentifierField.safeParse(vaeDispatch!.payload).success).toBe(true);
  });

  it('should include hash and type in Qwen3 encoder payload', () => {
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockAnimaMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    expect(qwen3Dispatch!.payload).toMatchObject({
      key: mockAnimaQwen3Encoder.key,
      hash: mockAnimaQwen3Encoder.hash,
      name: mockAnimaQwen3Encoder.name,
      base: mockAnimaQwen3Encoder.base,
      type: mockAnimaQwen3Encoder.type,
    });
  });

  it('should not dispatch encoder defaults when Anima models are already set', () => {
    const existingQwen3 = { key: 'existing', hash: 'h', name: 'Existing', base: 'any', type: 'qwen3_encoder' };
    const existingVae = { key: 'existing-vae', hash: 'h', name: 'Existing VAE', base: 'anima', type: 'vae' };

    const state = buildMockState({
      model: mockFluxMainModel,
      animaQwen3EncoderModel: existingQwen3,
      animaVaeModel: existingVae,
    });

    const action = modelSelected(zParameterModel.parse(mockAnimaMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    // Should NOT dispatch any encoder model selections since they're already set
    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    expect(qwen3Dispatch).toBeUndefined();
    expect(vaeDispatch).toBeUndefined();
  });

  it('should not dispatch encoder defaults when no encoder models are available', () => {
    mockSelectAnimaQwen3EncoderModels.mockReturnValue([]);
    mockSelectAnimaVAEModels.mockReturnValue([]);

    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockAnimaMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    expect(qwen3Dispatch).toBeUndefined();
    expect(vaeDispatch).toBeUndefined();
  });

  it('should clear Anima models when switching away from Anima', () => {
    const existingQwen3 = { key: 'existing', hash: 'h', name: 'Existing', base: 'any', type: 'qwen3_encoder' };
    const existingVae = { key: 'existing-vae', hash: 'h', name: 'Existing VAE', base: 'anima', type: 'vae' };

    const state = buildMockState({
      model: mockAnimaMainModel,
      animaQwen3EncoderModel: existingQwen3,
      animaVaeModel: existingVae,
    });

    const action = modelSelected(zParameterModel.parse(mockFluxMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    // Should dispatch null for both
    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    expect(qwen3Dispatch).toBeDefined();
    expect(qwen3Dispatch!.payload).toBeNull();
    expect(vaeDispatch).toBeDefined();
    expect(vaeDispatch!.payload).toBeNull();
  });
});

describe('modelSelected listener - Z-Image variant matching', () => {
  beforeEach(() => {
    dispatched.length = 0;
    mockDispatch.mockClear();
    mockSelectZImageDiffusersModels.mockReturnValue([mockZImageTurboDiffusers, mockZImageZbaseDiffusers]);
    mockSelectModelConfigsQuery.mockReturnValue({ data: { entities: {} } });
    mockModelConfigsSelectById.mockImplementation((_data, key) => {
      if (key === mockZImageTurboMain.key) {
        return mockZImageTurboMainConfig;
      }
      if (key === mockZImageZbaseMain.key) {
        return mockZImageZbaseMainConfig;
      }
      return undefined;
    });
  });

  it('should select turbo diffusers when switching from another base to Z-Image Turbo', () => {
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockZImageTurboMain));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const sourceDispatch = dispatched.find((a) => a.type === zImageQwen3SourceModelSelected.type);
    expect(sourceDispatch).toBeDefined();
    expect((sourceDispatch!.payload as { key: string }).key).toBe(mockZImageTurboDiffusers.key);
  });

  it('should select zbase diffusers when switching from another base to Z-Image Base', () => {
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockZImageZbaseMain));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const sourceDispatch = dispatched.find((a) => a.type === zImageQwen3SourceModelSelected.type);
    expect(sourceDispatch).toBeDefined();
    expect((sourceDispatch!.payload as { key: string }).key).toBe(mockZImageZbaseDiffusers.key);
  });

  it('should pick by variant, not list order — turbo wins even when zbase is first in the list', () => {
    // This is the regression: previously the listener took availableZImageDiffusers[0] unconditionally.
    mockSelectZImageDiffusersModels.mockReturnValue([mockZImageZbaseDiffusers, mockZImageTurboDiffusers]);
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockZImageTurboMain));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const sourceDispatch = dispatched.find((a) => a.type === zImageQwen3SourceModelSelected.type);
    expect(sourceDispatch).toBeDefined();
    expect((sourceDispatch!.payload as { key: string }).key).toBe(mockZImageTurboDiffusers.key);
  });

  it('should update source from turbo to zbase diffusers when switching variant within the same base', () => {
    const state = buildMockState({
      model: mockZImageTurboMain,
      zImageQwen3SourceModel: mockZImageTurboDiffusers,
    });
    const action = modelSelected(zParameterModel.parse(mockZImageZbaseMain));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const sourceDispatches = dispatched.filter((a) => a.type === zImageQwen3SourceModelSelected.type);
    expect(sourceDispatches).toHaveLength(1);
    expect((sourceDispatches[0]!.payload as { key: string }).key).toBe(mockZImageZbaseDiffusers.key);
  });

  it('should update source from zbase to turbo diffusers when switching variant within the same base', () => {
    const state = buildMockState({
      model: mockZImageZbaseMain,
      zImageQwen3SourceModel: mockZImageZbaseDiffusers,
    });
    const action = modelSelected(zParameterModel.parse(mockZImageTurboMain));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const sourceDispatches = dispatched.filter((a) => a.type === zImageQwen3SourceModelSelected.type);
    expect(sourceDispatches).toHaveLength(1);
    expect((sourceDispatches[0]!.payload as { key: string }).key).toBe(mockZImageTurboDiffusers.key);
  });

  it('should not update source when re-selecting the same Z-Image model key', () => {
    const state = buildMockState({
      model: mockZImageTurboMain,
      zImageQwen3SourceModel: mockZImageTurboDiffusers,
    });
    const action = modelSelected(zParameterModel.parse(mockZImageTurboMain));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const sourceDispatches = dispatched.filter((a) => a.type === zImageQwen3SourceModelSelected.type);
    expect(sourceDispatches).toHaveLength(0);
  });
});

describe('zModelIdentifierField schema validation', () => {
  it('should reject payloads missing hash and type', () => {
    const incomplete = { key: 'some-key', name: 'Some Model', base: 'any' };
    expect(zModelIdentifierField.safeParse(incomplete).success).toBe(false);
  });

  it('should accept payloads with all required fields', () => {
    const complete = { key: 'some-key', hash: 'some-hash', name: 'Some Model', base: 'any', type: 'qwen3_encoder' };
    expect(zModelIdentifierField.safeParse(complete).success).toBe(true);
  });
});
