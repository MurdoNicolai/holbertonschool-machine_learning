-- shows all tv show ratings
SELECT g.name AS genre, SUM(r.rate) AS rating_sum
FROM tv_genres AS g
INNER JOIN tv_show_genres AS tsg ON g.id = tsg.genre_id
INNER JOIN tv_shows AS ts ON tsg.show_id = ts.id
INNER JOIN tv_show_ratings AS r ON ts.id = r.show_id
GROUP BY g.name
ORDER BY rating_sum DESC;
