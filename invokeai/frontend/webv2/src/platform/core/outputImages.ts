/**
 * Shared adapter from backend invocation-output contracts to image names.
 *
 * This deliberately understands only the output shapes owned by the backend
 * contract. It is not a general object walker: metadata and invocation inputs
 * may contain images, but they are not outputs.
 */
type ImageNameVisitor = (imageName: string) => boolean;

const visitOutputImageNames = (output: unknown, visit: ImageNameVisitor): void => {
  const visitImageField = (value: unknown): boolean => {
    if (!value || typeof value !== 'object') {
      return true;
    }

    const imageName = (value as Record<string, unknown>).image_name;
    return typeof imageName === 'string' ? visit(imageName) : true;
  };

  const visitImageCollection = (value: unknown): boolean => {
    if (!Array.isArray(value)) {
      return true;
    }

    for (const image of value) {
      if (!visitImageField(image)) {
        return false;
      }
    }

    return true;
  };

  const visitWorkflowReturnValue = (value: unknown): boolean => {
    if (!value || typeof value !== 'object') {
      return true;
    }

    if (Array.isArray(value)) {
      return visitImageCollection(value);
    }

    const record = value as Record<string, unknown>;
    if (!visitImageField(record)) {
      return false;
    }

    if (record.type === 'image_output') {
      return visitImageField(record.image);
    }

    if (record.type === 'image_collection_output') {
      return visitImageCollection(record.collection);
    }

    if (
      'image' in record &&
      Object.keys(record).every((key) => key === 'image' || key === 'width' || key === 'height')
    ) {
      // Older workflow-return values omitted the output type for a single-image wrapper.
      return visitImageField(record.image);
    }

    if ('collection' in record && Object.keys(record).every((key) => key === 'collection')) {
      // Older workflow-return values omitted the output type for a collection wrapper.
      return visitImageCollection(record.collection);
    }

    return true;
  };

  if (!output || typeof output !== 'object' || Array.isArray(output)) {
    return;
  }

  const record = output as Record<string, unknown>;
  if (record.type === 'workflow_return_output' || 'values' in record) {
    if (record.values && typeof record.values === 'object' && !Array.isArray(record.values)) {
      for (const value of Object.values(record.values as Record<string, unknown>)) {
        if (!visitWorkflowReturnValue(value)) {
          return;
        }
      }
    }
    return;
  }

  if (record.type === 'image_output') {
    visitImageField(record.image);
    return;
  }

  if (record.type === 'image_collection_output') {
    visitImageCollection(record.collection);
    return;
  }

  if ('image' in record) {
    // Older queue results omitted the output type for single-image outputs.
    visitImageField(record.image);
    return;
  }

  if ('collection' in record) {
    // Older queue results omitted the output type for image collections.
    visitImageCollection(record.collection);
    return;
  }

  visitWorkflowReturnValue(record);
};

export const addOutputImageNames = (output: unknown, imageNames: Set<string>): void => {
  visitOutputImageNames(output, (imageName) => {
    imageNames.add(imageName);
    return true;
  });
};

export const getFirstOutputImageName = (output: unknown): string | undefined => {
  let firstImageName: string | undefined;

  visitOutputImageNames(output, (imageName) => {
    firstImageName = imageName;
    return false;
  });

  return firstImageName;
};

export const getOutputImageNames = (output: unknown): string[] => {
  const imageNames = new Set<string>();
  addOutputImageNames(output, imageNames);
  return [...imageNames];
};
