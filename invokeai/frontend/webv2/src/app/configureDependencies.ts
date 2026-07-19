import { identityTransportAuthAdapter } from '@features/identity';
import { configureHttpAuth } from '@platform/transport/http';

/** The App is the sole production adapter-selection and dependency-construction root. */
export const configureDependencies = (): void => {
  configureHttpAuth(identityTransportAuthAdapter);
};
