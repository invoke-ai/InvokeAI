import { z } from 'zod';

export const zPydanticValidationError = z.object({
  status: z.literal(422),
  error: z.object({
    detail: z.array(
      z.object({
        loc: z.array(z.string()),
        msg: z.string(),
        type: z.string(),
      })
    ),
  }),
});
