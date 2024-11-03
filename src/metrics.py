def contextual_precision(retrieval_context, true_context):

    if len(true_context) == 0 or len(retrieval_context) == 0:
        return 0

    cumulative_relevant = 0
    cumulative_precision = 0

    for k in range(1, len(retrieval_context)+1):
        is_relevant = retrieval_context[k-1] in true_context
        cumulative_relevant += is_relevant
        cumulative_precision += is_relevant * cumulative_relevant / k

    if cumulative_relevant == 0:
        return 0
    else:
        return cumulative_precision / cumulative_relevant


def contextual_recall(retrieval_context, true_context):

    if len(true_context) == 0 or len(retrieval_context) == 0:
        return 0

    retrieved_relevant = set(true_context).intersection(set(retrieval_context))
    return len(retrieved_relevant) / len(true_context)


def contextual_relevancy(retrieval_context, true_context):

    if len(true_context) == 0 or len(retrieval_context) == 0:
        return 0

    retrieved_relevant = set(true_context).intersection(set(retrieval_context))
    return len(retrieved_relevant) / len(retrieval_context)
