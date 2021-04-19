package com.example.cfm.ui.home;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.RecyclerView;

import com.example.cfm.R;
import com.example.spotify_framework.Song;
import com.example.spotify_framework.SongService;

import org.w3c.dom.Text;

import java.util.Collections;
import java.util.List;

public class SongsAdapter extends RecyclerView.Adapter<SongsAdapter.ViewHolder> {

    private LayoutInflater inflater;
    List<Song> data = Collections.EMPTY_LIST;

    public SongsAdapter(FragmentActivity activity, List<Song> data) {
        inflater = LayoutInflater.from(activity);
        this.data = data;
    }

    public int getItemCount() {
        return data.size();
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        View view = inflater.inflate(R.layout.song_recyclerview,viewGroup,false);
        ViewHolder holder = new ViewHolder(view);

        return holder;
    }

    public void onBindViewHolder(SongsAdapter.ViewHolder holder, int position) {
        System.out.println(data.size() + ":: Data Size");

        Song song = data.get(position);
        for(String s: song.getArtists())
            System.out.println(s);
        TextView nameView = holder.songNameView;
        TextView artistView = holder.songArtistView;
        String artists = song.getArtists().get(0);
        nameView.setText(song.getName());
        for(int i = 1; i < song.getArtists().size(); i++) {
            artists = artists + ", " + song.getArtists().get(i);
        }
        artistView.setText(artists);
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        private TextView songNameView;
        private TextView songArtistView;

        public ViewHolder(View view) {
            super(view);

            songNameView = (TextView)view.findViewById(R.id.song_name);
            songArtistView = (TextView)view.findViewById(R.id.song_artist);
        }

        public TextView getSongNameView() {
            return songNameView;
        }

        public TextView getSongArtistView() {
            return songArtistView;
        }
    }
}
