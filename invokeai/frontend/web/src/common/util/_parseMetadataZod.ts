/**
 * PARTIAL ZOD IMPLEMENTATION
 *
 * doesn't work well bc like most validators, zod is not built to skip invalid values.
 * it mostly works but just seems clearer and simpler to manually parse for now.
 *
 * in the future it would be really nice if we could use zod for some things:
 * - zodios (axios + zod): https://github.com/ecyrbe/zodios
 * - openapi to zodios: https://github.com/astahmer/openapi-zod-client
 */

// import { z } from 'zod';

// const zMetadataStringField = z.string();
// export type MetadataStringField = z.infer<typeof zMetadataStringField>;

// const zMetadataIntegerField = z.number().int();
// export type MetadataIntegerField = z.infer<typeof zMetadataIntegerField>;

// const zMetadataFloatField = z.number();
// export type MetadataFloatField = z.infer<typeof zMetadataFloatField>;

// const zMetadataBooleanField = z.boolean();
// export type MetadataBooleanField = z.infer<typeof zMetadataBooleanField>;

// const zMetadataImageField = z.object({
//   image_type: z.union([
//     z.literal('results'),
//     z.literal('uploads'),
//     z.literal('intermediates'),
//   ]),
//   image_name: z.string().min(1),
// });
// export type MetadataImageField = z.infer<typeof zMetadataImageField>;

// const zMetadataLatentsField = z.object({
//   latents_name: z.string().min(1),
// });
// export type MetadataLatentsField = z.infer<typeof zMetadataLatentsField>;

// /**
//  * zod Schema for any node field. Use a `transform()` to manually parse, skipping invalid values.
//  */
// const zAnyMetadataField = z.any().transform((val, ctx) => {
//   // Grab the field name from the path
//   const fieldName = String(ctx.path[ctx.path.length - 1]);

//   // `id` and `type` must be strings if they exist
//   if (['id', 'type'].includes(fieldName)) {
//     const reservedStringPropertyResult = zMetadataStringField.safeParse(val);
//     if (reservedStringPropertyResult.success) {
//       return reservedStringPropertyResult.data;
//     }

//     return;
//   }

//   // Parse the rest of the fields, only returning the data if the parsing is successful

//   const stringFieldResult = zMetadataStringField.safeParse(val);
//   if (stringFieldResult.success) {
//     return stringFieldResult.data;
//   }

//   const integerFieldResult = zMetadataIntegerField.safeParse(val);
//   if (integerFieldResult.success) {
//     return integerFieldResult.data;
//   }

//   const floatFieldResult = zMetadataFloatField.safeParse(val);
//   if (floatFieldResult.success) {
//     return floatFieldResult.data;
//   }

//   const booleanFieldResult = zMetadataBooleanField.safeParse(val);
//   if (booleanFieldResult.success) {
//     return booleanFieldResult.data;
//   }

//   const imageFieldResult = zMetadataImageField.safeParse(val);
//   if (imageFieldResult.success) {
//     return imageFieldResult.data;
//   }

//   const latentsFieldResult = zMetadataImageField.safeParse(val);
//   if (latentsFieldResult.success) {
//     return latentsFieldResult.data;
//   }
// });

// /**
//  * The node metadata schema.
//  */
// const zNodeMetadata = z.object({
//   session_id: z.string().min(1).optional(),
//   node: z.record(z.string().min(1), zAnyMetadataField).optional(),
// });

// export type NodeMetadata = z.infer<typeof zNodeMetadata>;

// const zMetadata = z.object({
//   invokeai: zNodeMetadata.optional(),
//   'sd-metadata': z.record(z.string().min(1), z.any()).optional(),
// });
// export type Metadata = z.infer<typeof zMetadata>;

// export const parseMetadata = (
//   metadata: Record<string, any>
// ): Metadata | undefined => {
//   const result = zMetadata.safeParse(metadata);
//   if (!result.success) {
//     console.log(result.error.issues);
//     return;
//   }

//   return result.data;
// };

export default {};
