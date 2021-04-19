package ui.notifications;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;


import com.example.cfm.R;
import spotify_framework.Playlist;

import java.util.List;
import java.util.Objects;



//import static com.example.flashcards.TaskFragment.GET_FROM_GALLERY; //not sure if it should be input activity

//this class creates views for data, and replaces the content of views when they are no longer available
//dang it i can't figure out how to make the inputviewHolder work as a static class instead of nonstatic class
public class PlaylistAdapter extends RecyclerView.Adapter<PlaylistAdapter.TaskViewHolder> {


    private List<Playlist> playlists; //cached copy of words
    private OnTaskItemClick onTaskItemClick;


    //obtains the layoutinflater from the given context. layoutinflater converts the xml file into its corresponding view
    PlaylistAdapter(Context frag, List<Playlist> tasks) {
        Objects.requireNonNull(tasks);
        this.playlists = tasks;

        // onTaskItemClick = (OnTaskItemClick) frag;
    }

    @Override
    @NonNull
    public TaskViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.playlist_recyclerview, parent, false);
        return new TaskViewHolder(itemView);
    }

    //TaskViewHolder is the current view of our cardview
    @Override
    public void onBindViewHolder(TaskViewHolder holder, int position) {
        Playlist current = playlists.get(position);
        //System.out.println("current task at position " + position + " is " + current.getTitle());
        holder.title.setText(current.getTitle());
        //MainActivity.makeCalendar(holder.date, this.mContext);
        holder.desc.setText(current.getDescription());
        holder.cover.setImageBitmap(current.getCover());
        holder.like_count.setText("liked songs: " + current.getLiked_songs());
        holder.dislike_count.setText(("disliked songs:" + current.getDisliked_songs()));
       /* switch (current.getPriority()) {
            case HIGH:
                holder.priority.setText("High Priority");
                break;
            case MED:
                holder.priority.setText("Medium Priority");
                break;
            case LOW:
                holder.priority.setText("Low Priority");
                break;
        } */
    }

    public void setPlaylists(List<Playlist> plist){
        this.playlists = plist;
        notifyDataSetChanged();
    }

    @Override
    public int getItemCount() {
        return playlists.size();
    }

    public Playlist getItemAt(int pos) {
        return playlists.get(pos);
    }

    public void addTasks(List<Playlist> newTasks) {
        playlists = newTasks;
        notifyDataSetChanged();
    }

    public interface OnTaskItemClick {
        void onTaskClick(int pos);
    }
    public List<Playlist> getPlaylists(){
        return playlists;
    }
    public class TaskViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        private TextView title;
        private TextView date;
        private TextView desc;
        private TextView reminder;
        private TextView priority;
        private ImageView cover;
        private TextView like_count;
        private TextView dislike_count;

        //this task view holds everything that should be contained in a playlist entry
        private TaskViewHolder(View itemView) {
            super(itemView);
            itemView.setOnClickListener(this);
            title = itemView.findViewById(R.id.plist_title);
            like_count = itemView.findViewById(R.id.plist_liked_song_count);
            dislike_count = itemView.findViewById(R.id.plist_disliked_songs_count);
            //date = itemView.findViewById(R.id.plist_disliked_songs_count);
            desc = itemView.findViewById(R.id.plist_description);
            cover = itemView.findViewById(R.id.plist_cover);
            //reminder = itemView.findViewById(R.id.);
            //priority = itemView.findViewById(R.id.priority_view);
        }

        @Override
        public void onClick(View v) {
            onTaskItemClick.onTaskClick(getAdapterPosition());
        }

    }
}

