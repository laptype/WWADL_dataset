# Original action_to_id mapping
action_to_id = {
    '伸懒腰': 0, '倒水': 1, '写字': 2, '切水果': 3, '吃水果': 4, '吃药': 5,
    '喝水': 6, '坐下': 7, '开关护眼灯': 8, '开关窗帘': 9, '开关窗户': 10, '打字': 11,
    '打开信封': 12, '扔垃圾': 13, '拿水果': 14, '捡东西': 15, '接电话': 16,
    '操作鼠标': 17, '擦桌子': 18, '板书': 19, '洗手': 20, '玩手机': 21,
    '看书': 22, '给植物浇水': 23, '走向床': 24, '走向椅子': 25, '走向橱柜': 26,
    '走向窗户': 27, '走向黑板': 28, '起床': 29, '起立': 30, '躺下': 31,
    '静止站立': 32, '静止躺着': 33
}

# Translation mapping
translations = {
    '伸懒腰': 'Stretching', '倒水': 'Pouring Water', '写字': 'Writing', '切水果': 'Cutting Fruit',
    '吃水果': 'Eating Fruit', '吃药': 'Taking Medicine', '喝水': 'Drinking Water', '坐下': 'Sitting Down',
    '开关护眼灯': 'Turning On/Off Eye Protection Lamp', '开关窗帘': 'Opening/Closing Curtains',
    '开关窗户': 'Opening/Closing Windows', '打字': 'Typing', '打开信封': 'Opening Envelope',
    '扔垃圾': 'Throwing Garbage', '拿水果': 'Picking Fruit', '捡东西': 'Picking Up Items', '接电话': 'Answering Phone',
    '操作鼠标': 'Using Mouse', '擦桌子': 'Wiping Table', '板书': 'Writing on Blackboard', '洗手': 'Washing Hands',
    '玩手机': 'Using Phone', '看书': 'Reading', '给植物浇水': 'Watering Plants', '走向床': 'Walking to Bed',
    '走向椅子': 'Walking to Chair', '走向橱柜': 'Walking to Cabinet', '走向窗户': 'Walking to Window',
    '走向黑板': 'Walking to Blackboard', '起床': 'Getting Out of Bed', '起立': 'Standing Up',
    '躺下': 'Lying Down', '静止站立': 'Standing Still', '静止躺着': 'Lying Still'
}

# Create id_to_action mapping
id_to_action = {v: translations[k] for k, v in action_to_id.items()}

# Output results
print("action_to_id:", action_to_id)
print("id_to_action:", id_to_action)