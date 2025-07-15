import { z } from 'zod/v4';

export const zPydanticValidationError = z.object({
  status: z.literal(422),
  data: z.object({
    detail: z.array(
      z.object({
        loc: z.array(z.string()),
        msg: z.string(),
        type: z.string(),
      })
    ),
  }),
});

export const zPydanticValidationErrorWithDetail = z.object({
  data: z.object({
    detail: z.string(),
  }),
});
