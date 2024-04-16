import type Konva from 'konva';
import { useCallback, useEffect, useState } from 'react';
import { Stage } from 'react-konva';

export const StageWrapper = () => {
  const [stage, setStage] = useState<Konva.Stage | null>(null);
  const stageRefCallback = useCallback(
    (el: Konva.Stage | null) => {
      setStage(el);
    },
    [setStage]
  );
  useEffect(() => {
    if (!stage) {
      return;
    }

    // do something with stage
  }, [stage]);

  return <Stage ref={stageRefCallback} />;
};
