package naiveautoml;

public class Phase1Results {

	private final FilteringResult filteringResult;
	private final RacePool racepool;

	public Phase1Results(final FilteringResult filteringResult, final RacePool racepool) {
		super();
		this.filteringResult = filteringResult;
		this.racepool = racepool;
	}

	public FilteringResult getFilteringResult() {
		return this.filteringResult;
	}

	public RacePool getRacepool() {
		return this.racepool;
	}
}
