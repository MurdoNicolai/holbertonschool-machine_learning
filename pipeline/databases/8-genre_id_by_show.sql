-- lists all shows that have at least one genre linked
SELECT ts.title, tsg.genre_id
FROM tv_shows AS ts
INNER JOIN tv_show_genres AS tsg ON ts.id = tsg.show_id
GROUP BY ts.title, tsg.genre_id
ORDER BY ts.title ASC, tsg.genre_id ASC;


