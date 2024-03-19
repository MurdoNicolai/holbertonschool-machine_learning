-- lists all showswithout a genre linked
SELECT ts.title AS title, NULL AS genre_id
FROM tv_shows AS ts
LEFT JOIN tv_show_genres AS tsg ON ts.id = tsg.show_id
WHERE tsg.genre_id IS NULL
ORDER BY ts.title ASC;
