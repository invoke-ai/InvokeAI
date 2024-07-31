import { useDestination } from 'features/parameters/hooks/useDestination';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { memo } from 'react';

type Props = {
  destination: InvokeTabName | undefined;
};

const Destination = (props: Props) => {
  useDestination(props.destination);
  return null;
};

export default memo(Destination);
