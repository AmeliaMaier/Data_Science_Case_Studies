def convert_to_numeric(catagorical):
    classes = catagorical.unique()
    classes_mapping = {cls: i for i, cls in enumerate(classes)}
    classes_inv_mapping = {i: cls for i, cls in enumerate(classes)}
    classes_numeric = catagorical.apply(lambda cls: classes_mapping[cls])
    return classes_numeric, classes_inv_mapping

fig, ax = plt.subplots(figsize=(6, 6))

catagorical = balance_non_zero["Student"]
y = balance_non_zero['Balance']

numeric, classes_mapping = convert_to_numeric(catagorical)

noise = np.random.uniform(-0.3, 0.3, size=len(catagorical))
ax.scatter(numeric + noise, y, color="grey", alpha=0.5)

box_data = list(y.groupby(catagorical))
ax.boxplot([data for _, data in box_data], positions=range(len(box_data)))
ax.set_xticks(list(classes_mapping))
ax.set_xticklabels(list(catagorical.unique()))

ax.set_xlabel("Student?")
ax.set_ylabel("Balance")
ax.set_title("Univariate Effect of Student? on Bank Balance")
