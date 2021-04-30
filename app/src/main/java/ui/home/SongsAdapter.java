package ui.home;

import android.app.Activity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.RecyclerView;

import com.example.cfm.R;
import com.squareup.picasso.Picasso;

import java.util.Collections;
import java.util.List;

import spotify_framework.Song;

public class SongsAdapter extends RecyclerView.Adapter<SongsAdapter.ViewHolder> {

    private LayoutInflater inflater;
    List<Song> data = Collections.EMPTY_LIST;
    Activity activity;

    public SongsAdapter(FragmentActivity activity, List<Song> data) {
        this.activity = activity;
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
        String artists;
        String name;
        String album;
        Song song = data.get(position);

        TextView nameView = holder.songNameView;
        TextView artistView = holder.songArtistView;
        ImageView albumView = holder.songAlbumArt;
        TextView albumName = holder.albumNameView;

        if(song.getName().length() > 20) {
            name = song.getName().substring(0,20) + "...";
        }
        else {
            name = song.getName();
        }

        if(song.getAlbum().length() > 20) {
            album = song.getAlbum().substring(0,20) + "...";
        }
        else {
            album = song.getAlbum();
        }
        Picasso.get().load(song.getImages().get(0)).into(albumView);

        artists = song.artistsToString(20);
        nameView.setText(name);
        artistView.setText("By: "+ artists);
        albumName.setText("On: "+ album);
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        private TextView songNameView;
        private TextView songArtistView;
        private TextView albumNameView;
        private ImageView songAlbumArt;

        public ViewHolder(View view) {
            super(view);

            songNameView = (TextView)view.findViewById(R.id.title);
            songArtistView = (TextView)view.findViewById(R.id.artist);
            songAlbumArt = (ImageView)view.findViewById(R.id.album_art);
            albumNameView = (TextView)view.findViewById(R.id.album);
        }

        public ImageView getSongAlbumArt() {return songAlbumArt;}

        public TextView getSongNameView() {
            return songNameView;
        }

        public TextView getSongArtistView() {
            return songArtistView;
        }

        public TextView getAlbumNameView() {return albumNameView;}
    }
}
