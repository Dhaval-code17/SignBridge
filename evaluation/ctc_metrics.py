def edit_distance(reference, hypothesis):
    """
    Calculate Levenshtein edit distance between two sequences.

    Returns:
        distance, substitutions, insertions, deletions
    """

    n = len(reference)
    m = len(hypothesis)

    # dp[i][j] stores:
    # minimum edits needed to convert
    # reference[:i] -> hypothesis[:j]
    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    # Track operation counts.
    operations = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        operations[i][0] = "delete"

    for j in range(1, m + 1):
        operations[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = "correct"

            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1

                best = min(
                    substitution,
                    insertion,
                    deletion
                )

                dp[i][j] = best

                if best == substitution:
                    operations[i][j] = "substitute"

                elif best == insertion:
                    operations[i][j] = "insert"

                else:
                    operations[i][j] = "delete"

    # Backtrack to count operations.
    i = n
    j = m

    substitutions = 0
    insertions = 0
    deletions = 0

    while i > 0 or j > 0:

        operation = operations[i][j]

        if operation == "correct":
            i -= 1
            j -= 1

        elif operation == "substitute":
            substitutions += 1
            i -= 1
            j -= 1

        elif operation == "insert":
            insertions += 1
            j -= 1

        elif operation == "delete":
            deletions += 1
            i -= 1

        else:
            break

    return (
        dp[n][m],
        substitutions,
        insertions,
        deletions
    )