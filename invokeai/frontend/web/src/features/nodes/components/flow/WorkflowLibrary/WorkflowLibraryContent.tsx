import { Flex } from '@chakra-ui/react';
import { memo, useState } from 'react';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import WorkflowLibraryCategories from './WorkflowLibraryCategories';
import WorkflowLibraryList from './WorkflowLibraryList';
import WorkflowLibraryPagination from './WorkflowLibraryPagination';
import { WorkflowCategory } from './types';

const PER_PAGE = 10;

const WorkflowLibraryContent = () => {
  const [page, setPage] = useState(0);
  const [category, setCategory] = useState<WorkflowCategory>('user');
  const { data } = useListWorkflowsQuery(
    {
      page,
      per_page: PER_PAGE,
    },
    { refetchOnMountOrArgChange: true }
  );

  if (!data) {
    return null;
  }

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
