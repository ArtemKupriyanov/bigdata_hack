def add_overall_offers_factors(train, test, subdf):
    weekday_count = subdf.groupby(['weekday_key'])['weekday_key'].agg({'weekday_key_orders' : 'size'}).reset_index()
    hour_count = subdf.groupby(['hour_key'])['hour_key'].agg({'hour_key_orders' : 'size'}).reset_index()
    weekday_hour_count = subdf.groupby(['weekday_key', 'hour_key'])[['hour_key', 'weekday_key']].agg({'weekday_hour_key_orders' : 'size'}).reset_index()
    train = train.merge(weekday_count, on='weekday_key')
    train = train.merge(hour_count, on='hour_key')
    train = train.merge(weekday_hour_count, on=['weekday_key', 'hour_key'])
    test = test.merge(weekday_count, on='weekday_key')
    test = test.merge(hour_count, on='hour_key')
    test = test.merge(weekday_hour_count, on=['weekday_key', 'hour_key'])
    
def add_mean_std_factors(train, test):
    for sl in [['driver_gk'], ['driver_gk', 'weekday_key'], ['driver_gk', 'hour_key']]:
        for column in ["hour_key", "driver_latitude", "driver_longitude", "origin_order_latitude", "origin_order_longitude", "distance_km"]:
            current_mean = train.groupby(sl)[column].agg({column + "_" + ' '.join(sl) + "_mean": "mean", column + "_" + ' '.join(sl) +"_std": "std"}).reset_index()
            train = train.merge(current_mean, on=sl)
            test = test.merge(current_mean, how="left", on=sl)