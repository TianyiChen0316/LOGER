
select count(distinct account.display_name) from account, so_user, badge b1, badge b2 where
account.website_url != '' and
account.id = so_user.account_id and

b1.site_id = so_user.site_id and
b1.user_id = so_user.id and
b1.name = 'Lifeboat' and

b2.site_id = so_user.site_id and
b2.user_id = so_user.id and
b2.name = 'Sheriff' and
b2.date > b1.date + '9 months'::interval
