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

const mockT5Encoder = {
  key: 't5-xxl-key',
  hash: 't5-xxl-hash',
  name: 'T5-XXL Encoder',
  base: 'any' as const,
  type: 't5_encoder' as const,
  format: 't5_encoder' as const,
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

const mockSelectT5EncoderModels = vi.fn((_state: unknown) => [mockT5Encoder]);

vi.mock('services/api/hooks/modelsByType', () => ({
  selectAnimaQwen3EncoderModels: (state: unknown) => mockSelectAnimaQwen3EncoderModels(state),
  selectAnimaVAEModels: (state: unknown) => mockSelectAnimaVAEModels(state),
  selectT5EncoderModels: (state: unknown) => mockSelectT5EncoderModels(state),
  selectQwen3EncoderModels: vi.fn(() => []),
  selectZImageDiffusersModels: vi.fn(() => []),
  selectFluxVAEModels: vi.fn(() => []),
  selectGlobalRefImageModels: vi.fn(() => []),
  selectRegionalRefImageModels: vi.fn(() => []),
}));

// Mock model configs adapter
vi.mock('services/api/endpoints/models', () => ({
  modelConfigsAdapterSelectors: { selectById: vi.fn() },
  selectModelConfigsQuery: vi.fn(() => ({ data: undefined })),
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
  animaT5EncoderModelSelected: { type: string };
  animaVaeModelSelected: { type: string };
};
const { animaQwen3EncoderModelSelected, animaT5EncoderModelSelected, animaVaeModelSelected } = paramsSliceActual;

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
      animaT5EncoderModel: null,
      animaScheduler: 'euler',
      kleinVaeModel: null,
      kleinQwen3EncoderModel: null,
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
    mockSelectT5EncoderModels.mockReturnValue([mockT5Encoder]);
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
    const t5Dispatch = dispatched.find((a) => a.type === animaT5EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    // All three should have been dispatched
    expect(qwen3Dispatch).toBeDefined();
    expect(t5Dispatch).toBeDefined();
    expect(vaeDispatch).toBeDefined();

    // The payloads must pass zModelIdentifierField validation (the actual schema used by reducers)
    expect(zModelIdentifierField.safeParse(qwen3Dispatch!.payload).success).toBe(true);
    expect(zModelIdentifierField.safeParse(t5Dispatch!.payload).success).toBe(true);
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

  it('should include hash and type in T5 encoder payload', () => {
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockAnimaMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    const t5Dispatch = dispatched.find((a) => a.type === animaT5EncoderModelSelected.type);
    expect(t5Dispatch!.payload).toMatchObject({
      key: mockT5Encoder.key,
      hash: mockT5Encoder.hash,
      name: mockT5Encoder.name,
      base: mockT5Encoder.base,
      type: mockT5Encoder.type,
    });
  });

  it('should not dispatch encoder defaults when Anima models are already set', () => {
    const existingQwen3 = { key: 'existing', hash: 'h', name: 'Existing', base: 'any', type: 'qwen3_encoder' };
    const existingT5 = { key: 'existing-t5', hash: 'h', name: 'Existing T5', base: 'any', type: 't5_encoder' };
    const existingVae = { key: 'existing-vae', hash: 'h', name: 'Existing VAE', base: 'anima', type: 'vae' };

    const state = buildMockState({
      model: mockFluxMainModel,
      animaQwen3EncoderModel: existingQwen3,
      animaT5EncoderModel: existingT5,
      animaVaeModel: existingVae,
    });

    const action = modelSelected(zParameterModel.parse(mockAnimaMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    // Should NOT dispatch any encoder model selections since they're already set
    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    const t5Dispatch = dispatched.find((a) => a.type === animaT5EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    expect(qwen3Dispatch).toBeUndefined();
    expect(t5Dispatch).toBeUndefined();
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
    const t5Dispatch = dispatched.find((a) => a.type === animaT5EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    expect(qwen3Dispatch).toBeUndefined();
    expect(t5Dispatch).toBeUndefined();
    expect(vaeDispatch).toBeUndefined();
  });

  it('should clear Anima models when switching away from Anima', () => {
    const existingQwen3 = { key: 'existing', hash: 'h', name: 'Existing', base: 'any', type: 'qwen3_encoder' };
    const existingT5 = { key: 'existing-t5', hash: 'h', name: 'Existing T5', base: 'any', type: 't5_encoder' };
    const existingVae = { key: 'existing-vae', hash: 'h', name: 'Existing VAE', base: 'anima', type: 'vae' };

    const state = buildMockState({
      model: mockAnimaMainModel,
      animaQwen3EncoderModel: existingQwen3,
      animaT5EncoderModel: existingT5,
      animaVaeModel: existingVae,
    });

    const action = modelSelected(zParameterModel.parse(mockFluxMainModel));

    capturedEffect!(action, {
      getState: () => state,
      dispatch: mockDispatch,
    });

    // Should dispatch null for all three
    const qwen3Dispatch = dispatched.find((a) => a.type === animaQwen3EncoderModelSelected.type);
    const t5Dispatch = dispatched.find((a) => a.type === animaT5EncoderModelSelected.type);
    const vaeDispatch = dispatched.find((a) => a.type === animaVaeModelSelected.type);

    expect(qwen3Dispatch).toBeDefined();
    expect(qwen3Dispatch!.payload).toBeNull();
    expect(t5Dispatch).toBeDefined();
    expect(t5Dispatch!.payload).toBeNull();
    expect(vaeDispatch).toBeDefined();
    expect(vaeDispatch!.payload).toBeNull();
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
