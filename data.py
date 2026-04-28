"""
Fed-ICL Replication — Data Module (v2 - Harder Task)
=====================================================
4-class news topic classification (AG News style).
This is much harder than binary sentiment — llama3 won't get 100% easily,
so you'll see real differences between baselines and Fed-ICL.
"""

import numpy as np
from config import SEED, DIRICHLET_ALPHA, NUM_CLIENTS, NUM_SERVER_QUERIES, EVAL_SIZE

np.random.seed(SEED)

LABEL_SPACE = ["world", "sports", "business", "science"]

RAW_DATA = [
    # ── WORLD (label=0) ──────────────────────────────────────
    ("Negotiations between the two nations broke down after diplomats failed to agree on border terms.", "world"),
    ("The United Nations called for an emergency session to address the growing humanitarian crisis.", "world"),
    ("Protesters gathered outside the parliament building demanding political reform.", "world"),
    ("A ceasefire agreement was reached after weeks of intense military conflict in the region.", "world"),
    ("The refugee camp now houses over fifty thousand displaced families from the conflict zone.", "world"),
    ("International observers raised concerns about irregularities in the recent presidential election.", "world"),
    ("The foreign minister announced the recall of ambassadors following a diplomatic dispute.", "world"),
    ("A peacekeeping force was deployed to the border area to prevent further escalation.", "world"),
    ("Trade sanctions were imposed on the country after allegations of human rights violations.", "world"),
    ("The summit concluded with a joint statement on climate cooperation between member states.", "world"),
    ("Rebel forces seized control of a key northern province after overnight clashes.", "world"),
    ("The prime minister survived a vote of no confidence by a narrow margin.", "world"),
    ("Aid agencies warned of a famine affecting millions in the drought-stricken eastern region.", "world"),
    ("A territorial dispute between neighbouring countries escalated with naval deployments.", "world"),
    ("The peace treaty signed last year appears to be unravelling amid fresh accusations.", "world"),
    ("Thousands marched through the capital calling for an end to government corruption.", "world"),
    ("A coalition of nations agreed to a joint military operation against insurgent groups.", "world"),
    ("Diplomatic ties between the two former allies were formally severed last week.", "world"),
    ("An earthquake devastated the coastal city leaving thousands homeless and infrastructure destroyed.", "world"),
    ("The international court issued an arrest warrant for the former head of state.", "world"),
    ("Cross-border tensions rose after a military incursion was reported along the disputed frontier.", "world"),
    ("The general assembly voted overwhelmingly in favour of the new disarmament resolution.", "world"),
    ("Ethnic violence displaced over a hundred thousand people from their homes this month.", "world"),
    ("The newly elected leader pledged sweeping democratic reforms in a televised address.", "world"),
    ("Flooding caused by the monsoon season has left entire villages submerged in the delta region.", "world"),
    ("A hostage situation at the embassy ended peacefully after twelve hours of negotiation.", "world"),
    ("The opposition party boycotted the election citing unfair media access and voter suppression.", "world"),
    ("A landmark agreement on nuclear nonproliferation was signed by five major powers.", "world"),
    ("Civil unrest spread across several provinces following the announcement of new austerity measures.", "world"),
    ("A volcano erupted on the island chain forcing the evacuation of nearby coastal communities.", "world"),
    ("The defence minister announced a significant increase in military spending for the coming year.", "world"),
    ("Migrant crossings reached record levels this quarter according to border agency figures.", "world"),
    ("A drone strike targeted a convoy in the northern highlands killing several suspected militants.", "world"),
    ("The president declared a state of emergency following widespread flooding in the south.", "world"),
    ("Relations between the eastern bloc nations and western alliance deteriorated sharply this month.", "world"),
    ("A coup attempt was foiled overnight with the arrest of several senior military officers.", "world"),
    ("The war crimes tribunal began hearings into atrocities committed during the decade-long conflict.", "world"),
    ("A cholera outbreak in the besieged city has overwhelmed the limited medical facilities.", "world"),

    # ── SPORTS (label=1) ──────────────────────────────────────
    ("The underdog team stunned the reigning champions with a last-minute goal to win the cup final.", "sports"),
    ("A world record was set in the hundred metre sprint at the international athletics championship.", "sports"),
    ("The tennis star announced her retirement after a career spanning two decades and multiple titles.", "sports"),
    ("Transfer negotiations for the star striker are expected to be completed before the window closes.", "sports"),
    ("The national team qualified for the tournament after a tense penalty shootout.", "sports"),
    ("An injury to the captain has thrown the squad's preparations for the final into disarray.", "sports"),
    ("The boxing champion defended his title with a unanimous decision over twelve rounds.", "sports"),
    ("The marathon runner collapsed just metres from the finish line in the extreme heat.", "sports"),
    ("A doping scandal has rocked the cycling world with several top riders facing suspensions.", "sports"),
    ("The swimming relay team broke the national record on their way to a silver medal.", "sports"),
    ("Fans clashed outside the stadium before the derby match leading to dozens of arrests.", "sports"),
    ("The rookie pitcher threw a complete game shutout in only his third major league start.", "sports"),
    ("The basketball team completed an unbeaten season with a dominant victory in the championship.", "sports"),
    ("An investigation has been launched into match-fixing allegations in the lower divisions.", "sports"),
    ("The golfer sank a remarkable forty-foot putt on the final hole to claim the tournament.", "sports"),
    ("The head coach was sacked after a run of six consecutive defeats in the league.", "sports"),
    ("A controversial offside decision denied the home side a crucial equaliser in stoppage time.", "sports"),
    ("The weightlifter earned a bronze medal despite competing with a shoulder injury.", "sports"),
    ("The Formula One driver secured pole position with a blistering qualifying lap.", "sports"),
    ("Youth academy graduates dominated the squad selection for the upcoming international fixtures.", "sports"),
    ("The gymnast scored a perfect ten on her floor routine to secure the all-around gold.", "sports"),
    ("A hat trick from the veteran forward sealed a comfortable away victory for the visitors.", "sports"),
    ("The cricket team declared on a mammoth total after a record-breaking double century.", "sports"),
    ("Ticket sales for the summer games exceeded expectations with most events already sold out.", "sports"),
    ("The ice hockey team rallied from two goals down to force overtime in the semifinal.", "sports"),
    ("A new coaching appointment has been confirmed following weeks of speculation in the media.", "sports"),
    ("The sprinter tested positive for a banned substance and faces a four-year competition ban.", "sports"),
    ("The sailing crew overcame difficult conditions to win the transoceanic race by a narrow margin.", "sports"),
    ("The rugby side suffered a heavy defeat in the opening round of the six nations championship.", "sports"),
    ("Stadium expansion plans were approved to increase capacity ahead of the continental tournament.", "sports"),
    ("The decathlete earned a personal best score to take the lead heading into the final event.", "sports"),
    ("The jockey guided the outsider to a surprise victory in the prestigious flat race.", "sports"),
    ("The volleyball team celebrated promotion after winning their final regular season match.", "sports"),
    ("Contract talks between the midfielder and the club have reportedly stalled over wage demands.", "sports"),
    ("The alpine skier clocked the fastest time in the downhill to add to her slalom gold.", "sports"),
    ("A bench-clearing brawl erupted in the ninth inning after a controversial hit-by-pitch call.", "sports"),
    ("The fencing champion earned her third consecutive title with a dominant final bout performance.", "sports"),
    ("The manager praised the team's defensive discipline after a hard-fought draw on the road.", "sports"),

    # ── BUSINESS (label=2) ────────────────────────────────────
    ("Shares in the technology company surged after quarterly earnings exceeded analyst expectations.", "business"),
    ("The central bank raised interest rates for the fourth consecutive time this year.", "business"),
    ("A major merger between two pharmaceutical giants was approved by regulators on Tuesday.", "business"),
    ("Unemployment figures fell to a ten-year low according to the latest government statistics.", "business"),
    ("The retail chain announced plans to close two hundred stores nationwide over the next year.", "business"),
    ("Oil prices dropped sharply following reports of increased output from producer nations.", "business"),
    ("The startup secured fifty million in Series B funding to expand into European markets.", "business"),
    ("Consumer confidence declined for the third straight month amid persistent inflation concerns.", "business"),
    ("The airline filed for bankruptcy protection after failing to restructure its mounting debts.", "business"),
    ("A hostile takeover bid for the media conglomerate was rejected by the board of directors.", "business"),
    ("The housing market showed signs of cooling as mortgage applications fell sharply.", "business"),
    ("Trade deficit figures widened as imports outpaced exports for the sixth consecutive quarter.", "business"),
    ("The automaker recalled half a million vehicles due to a defect in the braking system.", "business"),
    ("Quarterly GDP growth slowed to just half a percent raising fears of a potential recession.", "business"),
    ("The chief executive resigned amid an accounting scandal that wiped billions off the share price.", "business"),
    ("A landmark antitrust case against the search engine company began in federal court.", "business"),
    ("The cryptocurrency exchange halted withdrawals after a suspected security breach.", "business"),
    ("Inflation data came in higher than expected putting pressure on the central bank to act.", "business"),
    ("The insurance giant reported record profits driven by strong growth in its life policies division.", "business"),
    ("Factory output expanded for the first time in four months signalling a tentative recovery.", "business"),
    ("Supply chain disruptions continued to affect production at several major electronics manufacturers.", "business"),
    ("The hedge fund disclosed a significant short position in the struggling department store chain.", "business"),
    ("Bond yields climbed to their highest level in a decade as investors sold government debt.", "business"),
    ("The telecommunications provider announced a massive infrastructure investment in rural broadband.", "business"),
    ("Retail sales over the holiday period exceeded forecasts boosted by aggressive online discounts.", "business"),
    ("The banking sector faced renewed pressure after credit rating agencies downgraded several lenders.", "business"),
    ("A class action lawsuit was filed against the food manufacturer over misleading health claims.", "business"),
    ("The logistics company reported a surge in demand driven by the growth of online shopping.", "business"),
    ("Wage growth stalled despite a tight labour market puzzling economists and policymakers.", "business"),
    ("The energy company announced a pivot to renewables with plans to phase out coal by the decade end.", "business"),
    ("The stock exchange experienced its worst single-day decline since the financial crisis.", "business"),
    ("The conglomerate spun off its healthcare division into a separately listed public company.", "business"),
    ("A new trade agreement between the two largest economies sent markets to fresh highs.", "business"),
    ("The private equity firm acquired the struggling supermarket chain for an undisclosed sum.", "business"),
    ("Consumer spending slowed as households tightened budgets in response to rising living costs.", "business"),
    ("The venture capital firm closed its largest ever fund at two billion dollars.", "business"),
    ("The property developer defaulted on its bond payments sparking concerns across the sector.", "business"),
    ("The fintech company's initial public offering was the largest technology listing of the year.", "business"),

    # ── SCIENCE (label=3) ─────────────────────────────────────
    ("Researchers discovered a new species of deep-sea fish living near hydrothermal vents.", "science"),
    ("The space agency confirmed the successful landing of its rover on the surface of Mars.", "science"),
    ("A clinical trial showed promising results for a new treatment targeting drug-resistant bacteria.", "science"),
    ("Scientists observed gravitational waves from the merger of two neutron stars.", "science"),
    ("A breakthrough in quantum computing allowed researchers to solve a problem in minutes that would take classical computers millennia.", "science"),
    ("The fossil of an ancient marine reptile was unearthed in a limestone quarry in the midlands.", "science"),
    ("A new study linked microplastic contamination in oceans to declining reproductive rates in seabirds.", "science"),
    ("Astronomers identified a potentially habitable exoplanet orbiting a nearby red dwarf star.", "science"),
    ("Gene editing techniques were used to correct a hereditary condition in laboratory mice.", "science"),
    ("The particle accelerator detected an anomalous signal that could indicate new physics.", "science"),
    ("A vaccine candidate for malaria entered its final stage of human trials.", "science"),
    ("Satellite data revealed that the ice sheet lost mass at an accelerating rate over the past decade.", "science"),
    ("The telescope captured the most detailed image ever taken of a distant galaxy cluster.", "science"),
    ("Researchers developed a biodegradable plastic made entirely from plant-based materials.", "science"),
    ("A long-duration crewed mission to the space station set a new record for continuous habitation.", "science"),
    ("The genome sequencing project mapped the complete DNA of a high-altitude adapted population.", "science"),
    ("An artificial intelligence system outperformed human experts in diagnosing retinal disease from scans.", "science"),
    ("A coral reef restoration project showed encouraging signs of recovery after three years.", "science"),
    ("Nuclear fusion experiments achieved a net energy gain for the first time at the research facility.", "science"),
    ("The archaeological dig uncovered tools dating back over forty thousand years in the cave system.", "science"),
    ("Climate models predicted a significant shift in monsoon patterns over the coming decades.", "science"),
    ("The neuroscience team identified a neural pathway linked to habit formation in primates.", "science"),
    ("A new antibiotic compound was isolated from soil bacteria collected in a tropical rainforest.", "science"),
    ("The lunar sample return mission brought back material from a previously unexplored region.", "science"),
    ("Stem cell therapy restored partial mobility in patients with severe spinal cord injuries.", "science"),
    ("The oceanographic survey discovered an underwater mountain range in the southern ocean.", "science"),
    ("A wearable biosensor was developed that continuously monitors blood glucose without needles.", "science"),
    ("The superconducting material exhibited zero resistance at a temperature above minus fifty degrees.", "science"),
    ("Researchers demonstrated wireless power transmission over a distance of several kilometres.", "science"),
    ("An extinct flower was successfully grown from seeds preserved in permafrost for thirty thousand years.", "science"),
    ("The robotic spacecraft completed its flyby of the outer planet capturing unprecedented images.", "science"),
    ("A paper published this week reported the synthesis of a high-efficiency perovskite solar cell.", "science"),
    ("Brain-computer interface trials enabled paralysed patients to control a cursor with their thoughts.", "science"),
    ("The algal bloom monitoring system used satellite imagery to predict outbreaks weeks in advance.", "science"),
    ("The CRISPR gene drive technique was tested in mosquitoes to reduce malaria transmission.", "science"),
    ("A three-dimensional bioprinted organ was successfully transplanted into an animal model.", "science"),
    ("Paleontologists described a new dinosaur species from fossils found in the Patagonian desert.", "science"),
    ("Dark matter detection experiments reported an unexplained excess of events at low energies.", "science"),
]


def get_label_id(label: str) -> int:
    return LABEL_SPACE.index(label)


def partition_data_dirichlet(data: list, num_clients: int, alpha: float):
    labels = np.array([get_label_id(d[1]) for d in data])
    num_classes = len(LABEL_SPACE)
    client_data = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(class_indices)).astype(int)
        # Ensure at least 1 example per client per class when possible
        for k in range(num_clients):
            if counts[k] == 0 and len(class_indices) > num_clients:
                counts[k] = 1
        # Adjust first client to match total
        counts[0] += len(class_indices) - counts.sum()
        counts[0] = max(counts[0], 0)

        start = 0
        for k in range(num_clients):
            end = start + counts[k]
            for idx in class_indices[start:end]:
                client_data[k].append(data[idx])
            start = end

    for k in range(num_clients):
        np.random.shuffle(client_data[k])

    return client_data


def prepare_experiment():
    data = RAW_DATA.copy()
    np.random.shuffle(data)

    eval_set = data[:EVAL_SIZE]
    server_queries = data[EVAL_SIZE:EVAL_SIZE + NUM_SERVER_QUERIES]
    client_pool = data[EVAL_SIZE + NUM_SERVER_QUERIES:]

    client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    print("=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Total examples:    {len(RAW_DATA)}")
    print(f"  Label space:       {LABEL_SPACE}")
    print(f"  Evaluation set:    {len(eval_set)}")
    print(f"  Server queries:    {len(server_queries)}")
    print(f"  Client pool:       {len(client_pool)}")
    print(f"  Dirichlet alpha:   {DIRICHLET_ALPHA}")
    print()

    for k, cd in enumerate(client_datasets):
        counts = {l: sum(1 for _, lab in cd if lab == l) for l in LABEL_SPACE}
        total = len(cd)
        dist = ", ".join(f"{l}={counts[l]}" for l in LABEL_SPACE)
        print(f"  Client {k}: {total} examples ({dist})")

    print("=" * 60)
    return server_queries, client_datasets, eval_set


if __name__ == "__main__":
    sq, cd, ev = prepare_experiment()
    print(f"\nSample server query: {sq[0]}")
    print(f"Sample eval item:   {ev[0]}")
