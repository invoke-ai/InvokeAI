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

const mockKrea2MainModel = {
  key: 'krea2-main-key',
  hash: 'krea2-main-hash',
  name: 'Krea-2 Turbo',
  base: 'krea-2' as const,
  type: 'main' as const,
};

// Krea-2 borrows the Qwen-Image VAE and uses a standalone Qwen3-VL encoder for single-file / GGUF transformers.
const mockKrea2Vae = {
  key: 'krea2-vae-key',
  hash: 'krea2-vae-hash',
  name: 'Qwen Image VAE',
  base: 'qwen-image' as const,
  type: 'vae' as const,
  format: 'checkpoint' as const,
};

const mockKrea2Qwen3VlEncoder = {
  key: 'krea2-q3vl-key',
  hash: 'krea2-q3vl-hash',
  name: 'Qwen3-VL 4B Encoder',
  base: 'any' as const,
  type: 'qwen3_vl_encoder' as const,
  format: 'qwen3_vl_encoder' as const,
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

// Krea-2 standalone-component selectors (used only by the Krea-2 auto-select branch).
const mockSelectQwenImageVAEModels = vi.fn((_state: unknown) => [mockKrea2Vae]);
const mockSelectQwen3VLEncoderModels = vi.fn((_state: unknown) => [mockKrea2Qwen3VlEncoder]);

vi.mock('services/api/hooks/modelsByType', () => ({
  selectAnimaQwen3EncoderModels: (state: unknown) => mockSelectAnimaQwen3EncoderModels(state),
  selectAnimaVAEModels: (state: unknown) => mockSelectAnimaVAEModels(state),
  selectQwenImageVAEModels: (state: unknown) => mockSelectQwenImageVAEModels(state),
  selectQwen3VLEncoderModels: (state: unknown) => mockSelectQwen3VLEncoderModels(state),
  selectQwen3EncoderModels: vi.fn(() => []),
  selectZImageDiffusersModels: vi.fn(() => []),
  selectFluxVAEModels: vi.fn(() => []),
  selectGlobalRefImageModels: vi.fn(() => []),
  selectRegionalRefImageModels: vi.fn(() => []),
}));

// Mock model configs adapter. Routed through overridable fns so the Krea-2 tests can toggle the resolved
// model's format (diffusers vs. single-file/GGUF), which drives clear-vs-auto-select.
const mockSelectModelConfigsQuery = vi.fn((_state: unknown) => ({ data: undefined }) as { data: unknown });
const mockSelectModelById = vi.fn((_data: unknown, _key: string) => undefined as unknown);

vi.mock('services/api/endpoints/models', () => ({
  modelConfigsAdapterSelectors: { selectById: (data: unknown, key: string) => mockSelectModelById(data, key) },
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
  krea2VaeModelSelected: { type: string };
  krea2Qwen3VlEncoderModelSelected: { type: string };
};
const {
  animaQwen3EncoderModelSelected,
  animaVaeModelSelected,
  krea2VaeModelSelected,
  krea2Qwen3VlEncoderModelSelected,
} = paramsSliceActual;

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

describe('modelSelected listener - Krea-2 defaulting', () => {
  beforeEach(() => {
    dispatched.length = 0;
    mockDispatch.mockClear();
    // Standalone components installed by default; resolved model defaults to a non-diffusers (single-file /
    // GGUF) transformer, which is what triggers the auto-select branch.
    mockSelectQwenImageVAEModels.mockReturnValue([mockKrea2Vae]);
    mockSelectAnimaVAEModels.mockReturnValue([mockAnimaVAE]);
    mockSelectQwen3VLEncoderModels.mockReturnValue([mockKrea2Qwen3VlEncoder]);
    mockSelectModelConfigsQuery.mockReturnValue({ data: undefined });
    mockSelectModelById.mockReturnValue(undefined);
  });

  it('auto-selects a standalone VAE and Qwen3-VL encoder when switching to a single-file/GGUF Krea-2 model', () => {
    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockKrea2MainModel));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const vaeDispatch = dispatched.find((a) => a.type === krea2VaeModelSelected.type);
    const encoderDispatch = dispatched.find((a) => a.type === krea2Qwen3VlEncoderModelSelected.type);

    expect(vaeDispatch).toBeDefined();
    expect(encoderDispatch).toBeDefined();
    // The reducer parses payloads with zModelIdentifierField, so the dispatched values must be complete.
    expect(zModelIdentifierField.safeParse(vaeDispatch!.payload).success).toBe(true);
    expect(zModelIdentifierField.safeParse(encoderDispatch!.payload).success).toBe(true);
    expect(vaeDispatch!.payload).toMatchObject({ key: mockKrea2Vae.key, type: 'vae' });
    expect(encoderDispatch!.payload).toMatchObject({ key: mockKrea2Qwen3VlEncoder.key, type: 'qwen3_vl_encoder' });
  });

  it('falls back to an Anima-tagged VAE when no Qwen-Image VAE is installed', () => {
    mockSelectQwenImageVAEModels.mockReturnValue([]);
    mockSelectAnimaVAEModels.mockReturnValue([mockAnimaVAE]);

    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockKrea2MainModel));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const vaeDispatch = dispatched.find((a) => a.type === krea2VaeModelSelected.type);
    expect(vaeDispatch).toBeDefined();
    expect(vaeDispatch!.payload).toMatchObject({ key: mockAnimaVAE.key });
  });

  it('does not auto-select standalone components when none are installed', () => {
    // No Qwen-Image VAEs, no Anima VAEs (the fallback), no Qwen3-VL encoders.
    mockSelectQwenImageVAEModels.mockReturnValue([]);
    mockSelectAnimaVAEModels.mockReturnValue([]);
    mockSelectQwen3VLEncoderModels.mockReturnValue([]);

    const state = buildMockState({ model: mockFluxMainModel });
    const action = modelSelected(zParameterModel.parse(mockKrea2MainModel));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    expect(dispatched.find((a) => a.type === krea2VaeModelSelected.type)).toBeUndefined();
    expect(dispatched.find((a) => a.type === krea2Qwen3VlEncoderModelSelected.type)).toBeUndefined();
  });

  it('does not overwrite standalone components the user already selected', () => {
    const state = buildMockState({
      model: mockFluxMainModel,
      krea2VaeModel: { key: 'existing-vae', hash: 'h', name: 'Existing VAE', base: 'qwen-image', type: 'vae' },
      krea2Qwen3VlEncoderModel: {
        key: 'existing-enc',
        hash: 'h',
        name: 'Existing Enc',
        base: 'any',
        type: 'qwen3_vl_encoder',
      },
    });
    const action = modelSelected(zParameterModel.parse(mockKrea2MainModel));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    // Already set + non-diffusers -> nothing dispatched for the standalone slots.
    expect(dispatched.find((a) => a.type === krea2VaeModelSelected.type)).toBeUndefined();
    expect(dispatched.find((a) => a.type === krea2Qwen3VlEncoderModelSelected.type)).toBeUndefined();
  });

  it('clears stale standalone overrides when switching to a Diffusers Krea-2 model', () => {
    // A Diffusers pipeline bundles its own VAE + encoder, so any standalone overrides must be cleared.
    mockSelectModelConfigsQuery.mockReturnValue({ data: {} });
    mockSelectModelById.mockReturnValue({ format: 'diffusers' });

    const state = buildMockState({
      model: mockFluxMainModel,
      krea2VaeModel: { key: 'stale-vae', hash: 'h', name: 'Stale VAE', base: 'qwen-image', type: 'vae' },
      krea2Qwen3VlEncoderModel: {
        key: 'stale-enc',
        hash: 'h',
        name: 'Stale Enc',
        base: 'any',
        type: 'qwen3_vl_encoder',
      },
    });
    const action = modelSelected(zParameterModel.parse(mockKrea2MainModel));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const vaeDispatch = dispatched.find((a) => a.type === krea2VaeModelSelected.type);
    const encoderDispatch = dispatched.find((a) => a.type === krea2Qwen3VlEncoderModelSelected.type);
    expect(vaeDispatch).toBeDefined();
    expect(vaeDispatch!.payload).toBeNull();
    expect(encoderDispatch).toBeDefined();
    expect(encoderDispatch!.payload).toBeNull();
  });

  it('clears Krea-2 standalone components when switching away from Krea-2', () => {
    const state = buildMockState({
      model: mockKrea2MainModel,
      krea2VaeModel: { key: 'vae', hash: 'h', name: 'VAE', base: 'qwen-image', type: 'vae' },
      krea2Qwen3VlEncoderModel: { key: 'enc', hash: 'h', name: 'Enc', base: 'any', type: 'qwen3_vl_encoder' },
    });
    const action = modelSelected(zParameterModel.parse(mockFluxMainModel));

    capturedEffect!(action, { getState: () => state, dispatch: mockDispatch });

    const vaeDispatch = dispatched.find((a) => a.type === krea2VaeModelSelected.type);
    const encoderDispatch = dispatched.find((a) => a.type === krea2Qwen3VlEncoderModelSelected.type);
    expect(vaeDispatch).toBeDefined();
    expect(vaeDispatch!.payload).toBeNull();
    expect(encoderDispatch).toBeDefined();
    expect(encoderDispatch!.payload).toBeNull();
  });
});
