-- ============================================
-- Task 1: Find routes with at most 2 changes
-- ============================================

-- Direct trains (0 changes)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    NULL AS Change1,
    NULL AS Train2,
    NULL AS Change2,
    NULL AS Train3
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.station = t2.station AND s2.train = s1.train AND s2.arr IS NOT NULL
WHERE t1.station != t2.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))

UNION

-- One change (1 change)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    s2.station AS Change1,
    s3.train AS Train2,
    NULL AS Change2,
    NULL AS Train3
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.train = s1.train AND s2.arr IS NOT NULL
JOIN schedule s3 ON s3.station = s2.station AND s3.dep IS NOT NULL AND s3.train != s1.train
JOIN schedule s4 ON s4.station = t2.station AND s4.train = s3.train AND s4.arr IS NOT NULL
WHERE t1.station != t2.station
    AND t1.station != s2.station
    AND t2.station != s2.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))
    AND (s3.depdate > s2.arrdate 
         OR (s3.depdate = s2.arrdate AND s3.dep >= s2.arr))
    AND (s4.arrdate > s3.depdate 
         OR (s4.arrdate = s3.depdate AND s4.arr > s3.dep))
    -- Ensure no direct train exists
    AND NOT EXISTS (
        SELECT 1 
        FROM schedule d1 
        JOIN schedule d2 ON d2.train = d1.train
        WHERE d1.station = t1.station AND d1.dep IS NOT NULL
            AND d2.station = t2.station AND d2.arr IS NOT NULL
            AND (d2.arrdate > d1.depdate 
                 OR (d2.arrdate = d1.depdate AND d2.arr > d1.dep))
    )

UNION

-- Two changes (2 changes)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    s2.station AS Change1,
    s3.train AS Train2,
    s4.station AS Change2,
    s5.train AS Train3
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.train = s1.train AND s2.arr IS NOT NULL
JOIN schedule s3 ON s3.station = s2.station AND s3.dep IS NOT NULL AND s3.train != s1.train
JOIN schedule s4 ON s4.train = s3.train AND s4.arr IS NOT NULL
JOIN schedule s5 ON s5.station = s4.station AND s5.dep IS NOT NULL 
    AND s5.train != s3.train AND s5.train != s1.train
JOIN schedule s6 ON s6.station = t2.station AND s6.train = s5.train AND s6.arr IS NOT NULL
WHERE t1.station != t2.station
    AND t1.station != s2.station
    AND t1.station != s4.station
    AND t2.station != s2.station
    AND t2.station != s4.station
    AND s2.station != s4.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))
    AND (s3.depdate > s2.arrdate 
         OR (s3.depdate = s2.arrdate AND s3.dep >= s2.arr))
    AND (s4.arrdate > s3.depdate 
         OR (s4.arrdate = s3.depdate AND s4.arr > s3.dep))
    AND (s5.depdate > s4.arrdate 
         OR (s5.depdate = s4.arrdate AND s5.dep >= s4.arr))
    AND (s6.arrdate > s5.depdate 
         OR (s6.arrdate = s5.depdate AND s6.arr > s5.dep))
    -- Ensure no direct train exists
    AND NOT EXISTS (
        SELECT 1 
        FROM schedule d1 
        JOIN schedule d2 ON d2.train = d1.train
        WHERE d1.station = t1.station AND d1.dep IS NOT NULL
            AND d2.station = t2.station AND d2.arr IS NOT NULL
            AND (d2.arrdate > d1.depdate 
                 OR (d2.arrdate = d1.depdate AND d2.arr > d1.dep))
    )
    -- Ensure no one-change route exists
    AND NOT EXISTS (
        SELECT 1
        FROM schedule o1
        JOIN schedule o2 ON o2.train = o1.train AND o2.arr IS NOT NULL
        JOIN schedule o3 ON o3.station = o2.station AND o3.dep IS NOT NULL AND o3.train != o1.train
        JOIN schedule o4 ON o4.station = t2.station AND o4.train = o3.train AND o4.arr IS NOT NULL
        WHERE o1.station = t1.station AND o1.dep IS NOT NULL
            AND (o2.arrdate > o1.depdate 
                 OR (o2.arrdate = o1.depdate AND o2.arr > o1.dep))
            AND (o3.depdate > o2.arrdate 
                 OR (o3.depdate = o2.arrdate AND o3.dep >= o2.arr))
            AND (o4.arrdate > o3.depdate 
                 OR (o4.arrdate = o3.depdate AND o4.arr > o3.dep))
    )
ORDER BY StationA, StationB;

-- ============================================
-- Task 2: Find routes with at most 3 changes
-- ============================================

-- Direct trains (0 changes)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    NULL AS Change1,
    NULL AS Train2,
    NULL AS Change2,
    NULL AS Train3,
    NULL AS Change3,
    NULL AS Train4
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.station = t2.station AND s2.train = s1.train AND s2.arr IS NOT NULL
WHERE t1.station != t2.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))

UNION

-- One change (1 change)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    s2.station AS Change1,
    s3.train AS Train2,
    NULL AS Change2,
    NULL AS Train3,
    NULL AS Change3,
    NULL AS Train4
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.train = s1.train AND s2.arr IS NOT NULL
JOIN schedule s3 ON s3.station = s2.station AND s3.dep IS NOT NULL AND s3.train != s1.train
JOIN schedule s4 ON s4.station = t2.station AND s4.train = s3.train AND s4.arr IS NOT NULL
WHERE t1.station != t2.station
    AND t1.station != s2.station
    AND t2.station != s2.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))
    AND (s3.depdate > s2.arrdate 
         OR (s3.depdate = s2.arrdate AND s3.dep >= s2.arr))
    AND (s4.arrdate > s3.depdate 
         OR (s4.arrdate = s3.depdate AND s4.arr > s3.dep))
    AND NOT EXISTS (
        SELECT 1 FROM schedule d1 
        JOIN schedule d2 ON d2.train = d1.train
        WHERE d1.station = t1.station AND d1.dep IS NOT NULL
            AND d2.station = t2.station AND d2.arr IS NOT NULL
            AND (d2.arrdate > d1.depdate 
                 OR (d2.arrdate = d1.depdate AND d2.arr > d1.dep))
    )

UNION

-- Two changes (2 changes)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    s2.station AS Change1,
    s3.train AS Train2,
    s4.station AS Change2,
    s5.train AS Train3,
    NULL AS Change3,
    NULL AS Train4
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.train = s1.train AND s2.arr IS NOT NULL
JOIN schedule s3 ON s3.station = s2.station AND s3.dep IS NOT NULL AND s3.train != s1.train
JOIN schedule s4 ON s4.train = s3.train AND s4.arr IS NOT NULL
JOIN schedule s5 ON s5.station = s4.station AND s5.dep IS NOT NULL 
    AND s5.train != s3.train AND s5.train != s1.train
JOIN schedule s6 ON s6.station = t2.station AND s6.train = s5.train AND s6.arr IS NOT NULL
WHERE t1.station != t2.station
    AND t1.station != s2.station AND t1.station != s4.station
    AND t2.station != s2.station AND t2.station != s4.station
    AND s2.station != s4.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))
    AND (s3.depdate > s2.arrdate 
         OR (s3.depdate = s2.arrdate AND s3.dep >= s2.arr))
    AND (s4.arrdate > s3.depdate 
         OR (s4.arrdate = s3.depdate AND s4.arr > s3.dep))
    AND (s5.depdate > s4.arrdate 
         OR (s5.depdate = s4.arrdate AND s5.dep >= s4.arr))
    AND (s6.arrdate > s5.depdate 
         OR (s6.arrdate = s5.depdate AND s6.arr > s5.dep))
    AND NOT EXISTS (
        SELECT 1 FROM schedule d1 
        JOIN schedule d2 ON d2.train = d1.train
        WHERE d1.station = t1.station AND d1.dep IS NOT NULL
            AND d2.station = t2.station AND d2.arr IS NOT NULL
            AND (d2.arrdate > d1.depdate 
                 OR (d2.arrdate = d1.depdate AND d2.arr > d1.dep))
    )
    AND NOT EXISTS (
        SELECT 1 FROM schedule o1
        JOIN schedule o2 ON o2.train = o1.train AND o2.arr IS NOT NULL
        JOIN schedule o3 ON o3.station = o2.station AND o3.dep IS NOT NULL AND o3.train != o1.train
        JOIN schedule o4 ON o4.station = t2.station AND o4.train = o3.train AND o4.arr IS NOT NULL
        WHERE o1.station = t1.station AND o1.dep IS NOT NULL
            AND (o2.arrdate > o1.depdate 
                 OR (o2.arrdate = o1.depdate AND o2.arr > o1.dep))
            AND (o3.depdate > o2.arrdate 
                 OR (o3.depdate = o2.arrdate AND o3.dep >= o2.arr))
            AND (o4.arrdate > o3.depdate 
                 OR (o4.arrdate = o3.depdate AND o4.arr > o3.dep))
    )

UNION

-- Three changes (3 changes)
SELECT 
    t1.station AS StationA,
    t2.station AS StationB,
    s1.train AS Train1,
    s2.station AS Change1,
    s3.train AS Train2,
    s4.station AS Change2,
    s5.train AS Train3,
    s6.station AS Change3,
    s7.train AS Train4
FROM terminus t1
CROSS JOIN terminus t2
JOIN schedule s1 ON s1.station = t1.station AND s1.dep IS NOT NULL
JOIN schedule s2 ON s2.train = s1.train AND s2.arr IS NOT NULL
JOIN schedule s3 ON s3.station = s2.station AND s3.dep IS NOT NULL AND s3.train != s1.train
JOIN schedule s4 ON s4.train = s3.train AND s4.arr IS NOT NULL
JOIN schedule s5 ON s5.station = s4.station AND s5.dep IS NOT NULL 
    AND s5.train != s3.train AND s5.train != s1.train
JOIN schedule s6 ON s6.train = s5.train AND s6.arr IS NOT NULL
JOIN schedule s7 ON s7.station = s6.station AND s7.dep IS NOT NULL 
    AND s7.train != s5.train AND s7.train != s3.train AND s7.train != s1.train
JOIN schedule s8 ON s8.station = t2.station AND s8.train = s7.train AND s8.arr IS NOT NULL
WHERE t1.station != t2.station
    AND t1.station != s2.station AND t1.station != s4.station AND t1.station != s6.station
    AND t2.station != s2.station AND t2.station != s4.station AND t2.station != s6.station
    AND s2.station != s4.station AND s2.station != s6.station
    AND s4.station != s6.station
    AND (s2.arrdate > s1.depdate 
         OR (s2.arrdate = s1.depdate AND s2.arr > s1.dep))
    AND (s3.depdate > s2.arrdate 
         OR (s3.depdate = s2.arrdate AND s3.dep >= s2.arr))
    AND (s4.arrdate > s3.depdate 
         OR (s4.arrdate = s3.depdate AND s4.arr > s3.dep))
    AND (s5.depdate > s4.arrdate 
         OR (s5.depdate = s4.arrdate AND s5.dep >= s4.arr))
    AND (s6.arrdate > s5.depdate 
         OR (s6.arrdate = s5.depdate AND s6.arr > s5.dep))
    AND (s7.depdate > s6.arrdate 
         OR (s7.depdate = s6.arrdate AND s7.dep >= s6.arr))
    AND (s8.arrdate > s7.depdate 
         OR (s8.arrdate = s7.depdate AND s8.arr > s7.dep))
    -- Ensure no route with fewer changes exists
    AND NOT EXISTS (
        SELECT 1 FROM schedule d1 
        JOIN schedule d2 ON d2.train = d1.train
        WHERE d1.station = t1.station AND d1.dep IS NOT NULL
            AND d2.station = t2.station AND d2.arr IS NOT NULL
            AND (d2.arrdate > d1.depdate 
                 OR (d2.arrdate = d1.depdate AND d2.arr > d1.dep))
    )
    AND NOT EXISTS (
        SELECT 1 FROM schedule o1
        JOIN schedule o2 ON o2.train = o1.train AND o2.arr IS NOT NULL
        JOIN schedule o3 ON o3.station = o2.station AND o3.dep IS NOT NULL AND o3.train != o1.train
        JOIN schedule o4 ON o4.station = t2.station AND o4.train = o3.train AND o4.arr IS NOT NULL
        WHERE o1.station = t1.station AND o1.dep IS NOT NULL
            AND (o2.arrdate > o1.depdate 
                 OR (o2.arrdate = o1.depdate AND o2.arr > o1.dep))
            AND (o3.depdate > o2.arrdate 
                 OR (o3.depdate = o2.arrdate AND o3.dep >= o2.arr))
            AND (o4.arrdate > o3.depdate 
                 OR (o4.arrdate = o3.depdate AND o4.arr > o3.dep))
    )
    AND NOT EXISTS (
        SELECT 1 FROM schedule x1
        JOIN schedule x2 ON x2.train = x1.train AND x2.arr IS NOT NULL
        JOIN schedule x3 ON x3.station = x2.station AND x3.dep IS NOT NULL AND x3.train != x1.train
        JOIN schedule x4 ON x4.train = x3.train AND x4.arr IS NOT NULL
        JOIN schedule x5 ON x5.station = x4.station AND x5.dep IS NOT NULL 
            AND x5.train != x3.train AND x5.train != x1.train
        JOIN schedule x6 ON x6.station = t2.station AND x6.train = x5.train AND x6.arr IS NOT NULL
        WHERE x1.station = t1.station AND x1.dep IS NOT NULL
            AND (x2.arrdate > x1.depdate 
                 OR (x2.arrdate = x1.depdate AND x2.arr > x1.dep))
            AND (x3.depdate > x2.arrdate 
                 OR (x3.depdate = x2.arrdate AND x3.dep >= x2.arr))
            AND (x4.arrdate > x3.depdate 
                 OR (x4.arrdate = x3.depdate AND x4.arr > x3.dep))
            AND (x5.depdate > x4.arrdate 
                 OR (x5.depdate = x4.arrdate AND x5.dep >= x4.arr))
            AND (x6.arrdate > x5.depdate 
                 OR (x6.arrdate = x5.depdate AND x6.arr > x5.dep))
    )
ORDER BY StationA, StationB;

