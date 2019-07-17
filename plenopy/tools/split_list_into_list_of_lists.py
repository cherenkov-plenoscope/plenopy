def split_list_into_list_of_lists(events, num_events_in_job):
    """
    Splits a list into a list of sublists.
    Useful for distributing work onto several worker nodes.
    """
    num_events = len(events)
    jobs = []
    event_counter = 0
    while event_counter < num_events:
        job_event_counter = 0
        job = []
        while (
            event_counter < num_events and
            job_event_counter < num_events_in_job
        ):
            job.append(events[event_counter])
            job_event_counter += 1
            event_counter += 1
        jobs.append(job)
    return jobs
