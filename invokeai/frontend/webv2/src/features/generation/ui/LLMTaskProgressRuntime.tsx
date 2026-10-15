import { createLLMTaskProgressRuntime } from '@features/generation/data/llmTaskProgress';
import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';

/** Owns prompt-utility LLM progress listeners while the generation workspace is mounted. */
export const LLMTaskProgressRuntime = () => {
  useMountEffect(() => {
    const runtime = createLLMTaskProgressRuntime(socketHub);
    return runtime.dispose;
  });

  return null;
};
