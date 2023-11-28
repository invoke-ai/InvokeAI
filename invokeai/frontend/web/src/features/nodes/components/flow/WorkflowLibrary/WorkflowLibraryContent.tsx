import { Flex } from '@chakra-ui/react';
import { WorkflowCategory } from './types';
import { Dispatch, SetStateAction, memo } from 'react';
import { paths } from 'services/api/schema';
import WorkflowLibraryCategories from './WorkflowLibraryCategories';
import WorkflowLibraryPagination from './WorkflowLibraryPagination';
import WorkflowLibraryList from './WorkflowLibraryList';

type Props = {
  data: paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'];
  category: WorkflowCategory;
  setCategory: Dispatch<SetStateAction<WorkflowCategory>>;
  page: number;
  setPage: Dispatch<SetStateAction<number>>;
};

const WorkflowLibraryContent = ({
  data,
  category,
  setCategory,
  page,
  setPage,
}: Props) => {
  return (
    <Flex w="full" h="full" gap={2}>
      <WorkflowLibraryCategories
        category={category}
        setCategory={setCategory}
      />
      <Flex h="full" w="full" gap={2} flexDir="column">
        <WorkflowLibraryList data={data} />
        <WorkflowLibraryPagination data={data} page={page} setPage={setPage} />
      </Flex>
    </Flex>
  );
};

export default memo(WorkflowLibraryContent);
