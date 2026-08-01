interface OpenProjectSummary {
  id: string;
  name: string;
}

export const getProjectSwitcherSections = <TOpen extends OpenProjectSummary, TRecent extends OpenProjectSummary>(
  openProjects: TOpen[],
  libraryProjects: TRecent[],
  recentLimit: number
) => {
  const openProjectIds = new Set(openProjects.map((project) => project.id));

  return {
    open: openProjects,
    recent: libraryProjects.filter((project) => !openProjectIds.has(project.id)).slice(0, recentLimit),
  };
};
