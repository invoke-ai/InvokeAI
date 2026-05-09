import { z } from 'zod';

export const zHotkeysState = z.object({
  _version: z.literal(1),
  customHotkeys: z.record(z.string(), z.array(z.string())),
});

export type HotkeysState = z.infer<typeof zHotkeysState>;
