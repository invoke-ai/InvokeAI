import { Badge, Button } from '@chakra-ui/react';
import { PiBugBold } from 'react-icons/pi';

import { useWorkbench } from '../../WorkbenchContext';

export const DiagnosticsHeaderActions = () => {
  const { dispatch, state } = useWorkbench();

  return (
    <>
      {state.errorLog.length ? (
        <Badge colorPalette="red" size="xs">
          <PiBugBold />
          {state.errorLog.length}
        </Badge>
      ) : null}
      <Button
        disabled={state.errorLog.length === 0}
        size="2xs"
        variant="outline"
        onClick={() => dispatch({ type: 'clearErrorLog' })}
      >
        Clear
      </Button>
    </>
  );
};
