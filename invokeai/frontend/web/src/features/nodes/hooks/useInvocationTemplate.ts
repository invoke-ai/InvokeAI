import { useAppSelector } from 'app/storeHooks';
import { invocationTemplatesSelector } from '../store/selectors/invocationTemplatesSelector';

export const useGetInvocationTemplate = () => {
  const invocationTemplates = useAppSelector(invocationTemplatesSelector);

  return (invocationType: string) => {
    const template = invocationTemplates[invocationType];

    if (!template) {
      return;
    }

    return template;
  };
};
