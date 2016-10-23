def add2ax_object_distance_ruler(
	ax, 
	object_distance, 
	object_distance_min=0.0, 
	object_distance_max=10e3):
    ax.set_xlim([0, 1])
    ax.set_ylim([object_distance_min/1e3, object_distance_max/1e3])
    ax.yaxis.tick_left()
    ax.set_ylabel('object distance/km')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.plot(
    	[0, .5], 
    	[object_distance / 1e3, object_distance / 1e3], 
    	linewidth=5.0)
    ax.text(0.0, -1.0, format(object_distance / 1e3, '.2f') + r' km')
