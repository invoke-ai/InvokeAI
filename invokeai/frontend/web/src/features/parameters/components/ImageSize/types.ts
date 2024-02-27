import { z } from 'zod';

const zAspectRatioID = z.enum(['Free', '16:9', '3:2', '4:3', '1:1', '3:4', '2:3', '9:16']);
export type AspectRatioID = z.infer<typeof zAspectRatioID>;
export const isAspectRatioID = (v: string): v is AspectRatioID => zAspectRatioID.safeParse(v).success;

export type AspectRatioState = {
  id: AspectRatioID;
  value: number;
  isLocked: boolean;
};
