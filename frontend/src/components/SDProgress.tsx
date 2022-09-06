import { Progress } from '@chakra-ui/react';
import { useAppSelector } from '../app/hooks';
import { RootState } from '../app/store';

const SDProgress = () => {
    const { progress } = useAppSelector((state: RootState) => state.system);

    return <Progress size='xs' value={progress} />;
};

export default SDProgress;
