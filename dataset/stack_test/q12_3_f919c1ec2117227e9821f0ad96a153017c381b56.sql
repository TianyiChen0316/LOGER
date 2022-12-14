SELECT t1.name, count(*)
FROM
site as s,
so_user as u1,
question as q1,
answer as a1,
tag as t1,
tag_question as tq1
WHERE
q1.owner_user_id = u1.id
AND a1.question_id = q1.id
AND a1.owner_user_id = u1.id
AND s.site_id = q1.site_id
AND s.site_id = a1.site_id
AND s.site_id = u1.site_id
AND s.site_id = tq1.site_id
AND s.site_id = t1.site_id
AND q1.id = tq1.question_id
AND t1.id = tq1.tag_id
AND (s.site_name in ('academia','android','gamedev','gaming'))
AND (t1.name in ('3d','adb','career-path','conference','javascript','job-search','minecraft-pocket-edition','professors','ps3','shaders','terminology','united-states','writing'))
AND (q1.view_count >= 100)
AND (q1.view_count <= 100000)
AND (u1.reputation >= 0)
AND (u1.reputation <= 10)
GROUP BY t1.name